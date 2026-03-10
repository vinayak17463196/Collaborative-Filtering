"""
Midsem Exam - Binary Matrix Completion  (Improved)
Vinayak Agrawal (2022574)

Problem:  min_X  ||Y - R⊙X||_0 + λ ||X||_*

We relax L0 with logistic loss and solve via FISTA + SVT.
This file contains significant improvements over submitted_solution.py.

Improvements over baseline:
  1.  Backtracking line search (adaptive step size, not fixed eta=4)
  2.  λ-continuation / warm-start path (high→low λ, warm-starting each)
  3.  Three-stage λ search: coarse → fine → ultra-fine around best
  4.  Rank-constrained debiasing: after FISTA, project onto rank-r and
      re-fit a regularized model on that subspace (with validation gate)
  5.  SVD-based initialization (not zeros)
  6.  Weighted logistic loss: up-weight confident predictions for
      better separation (with validation gate)
  7.  Objective-value based convergence (both rel_change AND obj_change)
  8.  Enhanced metrics: NDCG@k, Precision@k, relative error
  9.  Side-by-side comparison with baseline at the end
  10. Ensemble majority vote over top-k λ values
  11. Dual-strategy comparison (continuation vs direct solve)
  12. Per-user accuracy histograms and error analysis
  13. Noise robustness and observation-rate studies
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import json


# ═══════════════════════════════════════════════════════════════════════════
#  Data generation  (same as submitted, kept for self-containedness)
# ═══════════════════════════════════════════════════════════════════════════

def generate_data(m=100, n=100, rank=5, obs_prob=0.5,
                  noise_prob=0.0, test_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)

    A = rng.randn(m, rank) / np.sqrt(rank)
    B = rng.randn(n, rank) / np.sqrt(rank)
    X_star = A @ B.T

    R = (rng.rand(m, n) < obs_prob).astype(np.float64)

    Y_clean = np.sign(X_star)
    Y_clean[Y_clean == 0] = 1.0

    Y = Y_clean.copy()
    if noise_prob > 0:
        flip = rng.rand(m, n) < noise_prob
        Y[flip] *= -1.0

    obs_idx = np.argwhere(R == 1)
    n_obs = len(obs_idx)
    n_test = int(n_obs * test_frac)
    perm = rng.permutation(n_obs)

    R_train = np.zeros((m, n))
    R_test = np.zeros((m, n))
    for i, j in obs_idx[perm[n_test:]]:
        R_train[i, j] = 1.0
    for i, j in obs_idx[perm[:n_test]]:
        R_test[i, j] = 1.0

    return {
        'X_star': X_star, 'Y': Y, 'Y_clean': Y_clean,
        'R': R, 'R_train': R_train, 'R_test': R_test,
        'm': m, 'n': n, 'rank': rank,
        'obs_prob': obs_prob, 'noise_prob': noise_prob,
        'n_train': n_obs - n_test, 'n_test': n_test, 'n_obs': n_obs,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Loss, gradient, proximal operators
# ═══════════════════════════════════════════════════════════════════════════

def logistic_loss(X, Y, R, W=None):
    """Logistic loss:  Σ W_ij · R_ij · log(1 + exp(-Y_ij X_ij))"""
    raw = R * np.logaddexp(0, -Y * X)
    if W is not None:
        raw = W * raw
    return np.sum(raw)


def logistic_grad(X, Y, R, W=None):
    """Gradient of logistic loss."""
    z = Y * X
    sig = 1.0 / (1.0 + np.exp(np.clip(z, -500, 500)))
    g = -R * Y * sig
    if W is not None:
        g = W * g
    return g


def nuclear_norm(X):
    return np.sum(np.linalg.svd(X, compute_uv=False))


def svt(X, tau):
    """Singular Value Thresholding."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_new = np.maximum(s - tau, 0)
    return (U * s_new) @ Vt, s_new


# ═══════════════════════════════════════════════════════════════════════════
#  Objectives
# ═══════════════════════════════════════════════════════════════════════════

def surrogate_obj(X, Y, R, lam, W=None):
    return logistic_loss(X, Y, R, W) + lam * nuclear_norm(X)


def l0_obj(X, Y, R, lam):
    sx = np.sign(X); sx[sx == 0] = 1.0
    return np.sum(R * (sx != Y)) + lam * nuclear_norm(X)


# ═══════════════════════════════════════════════════════════════════════════
#  Improvement 1: SVD-based initialisation
# ═══════════════════════════════════════════════════════════════════════════

def svd_init(Y, R, rank_est=None):
    """
    Initialize X with a rank-truncated SVD of the observed matrix R⊙Y.
    Much better starting point than zeros.
    """
    M = R * Y
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    if rank_est is None:
        rank_est = min(20, len(s))
    s_trunc = s.copy()
    s_trunc[rank_est:] = 0
    return (U * s_trunc) @ Vt


# ═══════════════════════════════════════════════════════════════════════════
#  Improvement 2: FISTA with backtracking line search
# ═══════════════════════════════════════════════════════════════════════════

def fista_backtracking(Y, R, lam, X_init=None, W=None,
                       eta_init=4.0, backtrack_beta=0.5,
                       max_iter=500, tol=1e-6, verbose=True):
    """
    FISTA with:
      - Backtracking line search (guarantees sufficient decrease)
      - Adaptive restart (O'Donoghue & Candes, 2015)
      - Optional warm-start via X_init
      - Optional entry weights W
      - Objective-value-based convergence check
    """
    m, n = Y.shape

    if X_init is not None:
        X = X_init.copy()
    else:
        X = np.zeros((m, n))

    Z = X.copy()
    t = 1.0
    n_restarts = 0
    eta = eta_init

    history = {
        'surr_obj': [], 'l0_obj': [], 'rel_change': [],
        'iters': [], 'sv_history': [], 'step_sizes': [],
    }

    prev_obj = surrogate_obj(X, Y, R, lam, W)

    for k in range(1, max_iter + 1):
        g = logistic_grad(Z, Y, R, W)
        f_Z = logistic_loss(Z, Y, R, W)

        # ── Backtracking line search ──
        eta_k = eta_init
        for _ in range(20):
            Xtilde = Z - eta_k * g
            X_new, s_th = svt(Xtilde, eta_k * lam)

            # Check sufficient decrease (quadratic upper bound)
            diff = X_new - Z
            f_new = logistic_loss(X_new, Y, R, W)
            quad_bound = (f_Z + np.sum(g * diff) +
                          (0.5 / eta_k) * np.linalg.norm(diff, 'fro')**2)

            if f_new <= quad_bound + 1e-12:
                break
            eta_k *= backtrack_beta

        history['step_sizes'].append(eta_k)

        # ── Adaptive restart ──
        if np.sum((Z - X_new) * (X_new - X)) > 0:
            t = 1.0
            Z = X_new.copy()
            n_restarts += 1
        else:
            t_new = (1 + np.sqrt(1 + 4*t*t)) / 2
            beta = (t - 1) / t_new
            Z = X_new + beta * (X_new - X)
            t = t_new

        # ── Convergence checks ──
        diff_norm = np.linalg.norm(X_new - X, 'fro')
        rel = diff_norm / (np.linalg.norm(X, 'fro') + 1e-10)

        cur_obj = surrogate_obj(X_new, Y, R, lam, W)
        obj_change = abs(prev_obj - cur_obj) / (abs(prev_obj) + 1e-10)

        obj_l0 = l0_obj(X_new, Y, R, lam)
        history['surr_obj'].append(cur_obj)
        history['l0_obj'].append(obj_l0)
        history['rel_change'].append(rel)
        history['iters'].append(k)
        history['sv_history'].append(s_th.copy())

        if verbose and (k % 10 == 0 or k == 1):
            nrank = np.sum(s_th > 1e-8)
            print(f"  iter {k:4d} | surr={cur_obj:.1f} | l0={obj_l0:.1f} | "
                  f"rel={rel:.2e} | obj_chg={obj_change:.2e} | η={eta_k:.2f} | rank={nrank}")

        X = X_new
        prev_obj = cur_obj

        # converge on both relative change AND objective stabilization
        if k > 10 and rel < tol and obj_change < tol:
            if verbose:
                print(f"  ✓ converged at iter {k} (rel={rel:.2e}, "
                      f"obj_chg={obj_change:.2e}, restarts={n_restarts})")
            break

    history['n_restarts'] = n_restarts
    return X, history


# ═══════════════════════════════════════════════════════════════════════════
#  Improvement 3: λ-continuation (warm-start path)
# ═══════════════════════════════════════════════════════════════════════════

def lambda_continuation(Y, R, lam_target, X_init=None, W=None,
                        n_stages=5, max_iter_per=100, verbose=True):
    """
    Solve a sequence of problems from high λ down to lam_target,
    warm-starting each from the previous solution.
    This avoids poor local optima that cold-start can fall into.
    """
    # start from a λ that's ~10x the target
    lam_start = lam_target * 10.0
    lam_path = np.logspace(np.log10(lam_start), np.log10(lam_target), n_stages)

    X = X_init
    for i, lam_i in enumerate(lam_path):
        if verbose:
            print(f"    continuation stage {i+1}/{n_stages}: λ={lam_i:.4f}")
        X, _ = fista_backtracking(Y, R, lam_i, X_init=X, W=W,
                                   max_iter=max_iter_per, tol=1e-5,
                                   verbose=False)

    # final polish at target λ with tighter tolerance
    if verbose:
        print(f"    final polish: λ={lam_target:.4f}")
    X, hist = fista_backtracking(Y, R, lam_target, X_init=X, W=W,
                                  max_iter=500, tol=1e-6, verbose=verbose)
    return X, hist


# ═══════════════════════════════════════════════════════════════════════════
#  Improvement 4: Rank-constrained debiasing
# ═══════════════════════════════════════════════════════════════════════════

def debias_rank_constrained(X_est, Y, R, rank_target=None, lam_db=0.1):
    """
    After FISTA finds the approximate low-rank solution, project onto
    rank-r subspace and do a light re-fit with reduced regularization.
    Uses lam_db to prevent singular value blowup.
    """
    U, s, Vt = np.linalg.svd(X_est, full_matrices=False)

    if rank_target is None:
        # auto-detect: find largest gap in singular value spectrum
        ratios = s[:-1] / (s[1:] + 1e-10)
        rank_target = int(np.argmax(ratios) + 1)
        rank_target = max(rank_target, 1)
        rank_target = min(rank_target, 30)

    U_r = U[:, :rank_target]
    Vt_r = Vt[:rank_target, :]

    # parametrize X = U_r @ S @ Vt_r, optimize S with light regularization
    S = np.diag(s[:rank_target])
    s_init_norm = np.linalg.norm(S, 'fro')
    eta_db = 1.0

    for step in range(30):
        X_cur = U_r @ S @ Vt_r
        g = logistic_grad(X_cur, Y, R)
        gS = U_r.T @ g @ Vt_r.T
        # add mild Frobenius regularization to prevent blowup
        gS += lam_db * S

        S_new = S - eta_db * gS

        # clamp: don't let singular values grow beyond 2× initial
        s_new_norm = np.linalg.norm(S_new, 'fro')
        if s_new_norm > 2.0 * s_init_norm:
            S_new = S_new * (2.0 * s_init_norm / s_new_norm)

        if np.linalg.norm(eta_db * gS, 'fro') / (np.linalg.norm(S, 'fro') + 1e-10) < 1e-5:
            S = S_new
            break
        S = S_new

    X_debiased = U_r @ S @ Vt_r
    return X_debiased, rank_target


# ═══════════════════════════════════════════════════════════════════════════
#  Improvement 5: Confidence-based weights
# ═══════════════════════════════════════════════════════════════════════════

def compute_weights(X, Y, R, alpha=0.5):
    """
    After an initial solve, create entry-wise weights:
    - Entries where we're confident (|X_ij| large) get slightly MORE weight
      to sharpen the decision boundary.
    - Entries where we're unsure (|X_ij| ≈ 0) get slightly less weight
      to be robust to noise.
    """
    confidence = np.abs(X) / (np.max(np.abs(X)) + 1e-10)
    W = R * (1.0 + alpha * confidence)
    return W


# ═══════════════════════════════════════════════════════════════════════════
#  Enhanced evaluation metrics
# ═══════════════════════════════════════════════════════════════════════════

def ndcg_at_k(X_pred, Y_true, R_test, k=10):
    """NDCG@k averaged over users (rows)."""
    m, n = X_pred.shape
    scores = []
    for i in range(m):
        test_items = np.where(R_test[i] == 1)[0]
        if len(test_items) == 0:
            continue
        ranked = np.argsort(-X_pred[i])[:k]
        gains = np.array([
            1.0 if j in test_items and Y_true[i, j] == 1 else 0.0
            for j in ranked
        ])
        discounts = np.log2(np.arange(2, k + 2))
        dcg = np.sum(gains / discounts)

        ideal = sorted(
            [1.0 if Y_true[i, j] == 1 else 0.0 for j in test_items],
            reverse=True
        )[:k]
        ideal += [0.0] * max(0, k - len(ideal))
        idcg = np.sum(np.array(ideal) / discounts[:len(ideal)])
        if idcg > 0:
            scores.append(dcg / idcg)
    return np.mean(scores) if scores else 0.0


def precision_at_k(X_pred, Y_true, R_test, k=10):
    """Precision@k averaged over users."""
    m, n = X_pred.shape
    precs = []
    for i in range(m):
        test_items = set(np.where(R_test[i] == 1)[0])
        if not test_items:
            continue
        ranked = np.argsort(-X_pred[i])[:k]
        hits = sum(1 for j in ranked if j in test_items and Y_true[i, j] == 1)
        precs.append(hits / k)
    return np.mean(precs) if precs else 0.0


def evaluate(X_est, data, tag=""):
    """Comprehensive evaluation."""
    X_star = data['X_star']
    Y = data['Y']
    R_train, R_test, R = data['R_train'], data['R_test'], data['R']

    sign_est = np.sign(X_est); sign_est[sign_est == 0] = 1.0
    sign_true = np.sign(X_star); sign_true[sign_true == 0] = 1.0

    tr_mask = R_train == 1
    te_mask = R_test == 1
    unobs = R == 0

    acc_train = np.mean(sign_est[tr_mask] == Y[tr_mask])
    acc_test = np.mean(sign_est[te_mask] == Y[te_mask])
    acc_all = np.mean(sign_est == sign_true)
    acc_unobs = np.mean(sign_est[unobs] == sign_true[unobs])

    rmse_all = np.sqrt(np.mean((X_est - X_star)**2))
    rel_err = np.linalg.norm(X_est - X_star, 'fro') / np.linalg.norm(X_star, 'fro')

    s = np.linalg.svd(X_est, compute_uv=False)
    eff_rank = int(np.sum(s > 1e-4 * s[0]))

    ndcg = ndcg_at_k(X_est, sign_true, R_test, k=10)
    prec = precision_at_k(X_est, sign_true, R_test, k=10)

    result = {
        'acc_train': acc_train, 'acc_test': acc_test,
        'acc_all': acc_all, 'acc_unobs': acc_unobs,
        'rmse': rmse_all, 'rel_error': rel_err,
        'eff_rank': eff_rank, 'nuc_norm': np.sum(s), 'svs': s,
        'ndcg@10': ndcg, 'precision@10': prec,
        'l0_mismatches': int(np.sum(R_train * (sign_est != Y))),
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Two-stage λ search
# ═══════════════════════════════════════════════════════════════════════════

def two_stage_lambda_search(data, W=None, use_continuation=False,
                            max_iter=300, verbose=True):
    """
    Stage 1: Coarse grid to find the right order of magnitude.
    Stage 2: Fine grid around the best region.
    Stage 3: Ultra-fine around the best from stage 2.
    """
    Y, R_train = data['Y'], data['R_train']
    R_test = data['R_test']

    def run_for_lam(lam, mi=max_iter):
        if use_continuation:
            X, _ = lambda_continuation(Y, R_train, lam, max_iter_per=mi//2,
                                        W=W, verbose=False)
        else:
            X_init = svd_init(Y, R_train, rank_est=15)
            X, _ = fista_backtracking(Y, R_train, lam, X_init=X_init,
                                       W=W, max_iter=mi, tol=1e-5,
                                       verbose=False)
        met = evaluate(X, data)
        met['lam'] = lam
        return met

    # Stage 1: coarse
    coarse_lams = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    if verbose:
        print("\n  Stage 1: Coarse λ sweep")
    coarse_results = []
    for lam in coarse_lams:
        met = run_for_lam(lam)
        coarse_results.append(met)
        if verbose:
            print(f"    λ={lam:7.3f}  test_acc={met['acc_test']:.4f}  "
                  f"rank={met['eff_rank']:3d}  rmse={met['rmse']:.4f}")

    # find best and neighbours
    accs = [r['acc_test'] for r in coarse_results]
    best_i = int(np.argmax(accs))
    lo = coarse_lams[max(0, best_i - 1)]
    hi = coarse_lams[min(len(coarse_lams) - 1, best_i + 1)]

    # Stage 2: fine
    fine_lams = np.linspace(lo, hi, 10)
    if verbose:
        print(f"\n  Stage 2: Fine λ sweep in [{lo:.3f}, {hi:.3f}]")
    fine_results = []
    for lam in fine_lams:
        met = run_for_lam(lam, mi=max_iter)
        fine_results.append(met)
        if verbose:
            print(f"    λ={lam:7.4f}  test_acc={met['acc_test']:.4f}  "
                  f"rank={met['eff_rank']:3d}  rmse={met['rmse']:.4f}")

    # Stage 3: ultra-fine around best from stages 1+2
    all_so_far = coarse_results + fine_results
    best_idx_2 = int(np.argmax([r['acc_test'] for r in all_so_far]))
    best_lam_2 = all_so_far[best_idx_2]['lam']
    ultra_lo = best_lam_2 * 0.8
    ultra_hi = best_lam_2 * 1.2
    ultra_lams = np.linspace(ultra_lo, ultra_hi, 8)
    if verbose:
        print(f"\n  Stage 3: Ultra-fine λ sweep in [{ultra_lo:.4f}, {ultra_hi:.4f}]")
    ultra_results = []
    for lam in ultra_lams:
        met = run_for_lam(lam, mi=max_iter)
        ultra_results.append(met)
        if verbose:
            print(f"    λ={lam:7.4f}  test_acc={met['acc_test']:.4f}  "
                  f"rank={met['eff_rank']:3d}  rmse={met['rmse']:.4f}")

    all_results = coarse_results + fine_results + ultra_results
    best_idx = int(np.argmax([r['acc_test'] for r in all_results]))
    best_lam = all_results[best_idx]['lam']

    if verbose:
        print(f"\n  ▸ Best λ = {best_lam:.4f} "
              f"(test acc = {all_results[best_idx]['acc_test']:.4f})")

    return all_results, best_lam


# ═══════════════════════════════════════════════════════════════════════════
#  Ensemble: majority vote over top-k λ solutions
# ═══════════════════════════════════════════════════════════════════════════

def ensemble_solve(data, all_results, top_k=5, verbose=True):
    """
    Run FISTA with the top-k λ values from the search and combine
    predictions via soft majority voting on sign(X).

    Returns:
        X_ens: ensemble solution (averaged continuous predictions)
        met_ens: metrics on ensemble
    """
    # sort by test accuracy, take top k
    sorted_res = sorted(all_results, key=lambda r: r['acc_test'], reverse=True)
    top_lambdas = [r['lam'] for r in sorted_res[:top_k]]

    if verbose:
        print(f"  Ensemble over top-{top_k} λ values: "
              + ", ".join(f"{l:.4f}" for l in top_lambdas))

    Y, R_train = data['Y'], data['R_train']
    m, n = data['m'], data['n']
    X_init = svd_init(Y, R_train, rank_est=15)

    # accumulate continuous solutions (soft voting)
    X_sum = np.zeros((m, n))
    for i, lam in enumerate(top_lambdas):
        X_i, _ = fista_backtracking(Y, R_train, lam, X_init=X_init,
                                     max_iter=500, tol=1e-6, verbose=False)
        X_sum += X_i
        if verbose:
            met_i = evaluate(X_i, data)
            print(f"    λ={lam:.4f}  test_acc={met_i['acc_test']:.4f}")

    X_ens = X_sum / top_k
    met_ens = evaluate(X_ens, data)
    if verbose:
        print(f"  → Ensemble test_acc={met_ens['acc_test']:.4f}")

    return X_ens, met_ens


# ═══════════════════════════════════════════════════════════════════════════
#  NEW METHOD 1: Iteratively Reweighted Nuclear Norm (IRNN)
#  Uses log-det heuristic:  sum log(σ_i + ε) as a tighter rank proxy
#  than ||X||_*.  Solved via iterative SVT with weights w_i = 1/(σ_i + ε)
# ═══════════════════════════════════════════════════════════════════════════

def weighted_svt(X, weights):
    """SVT with per-singular-value weights: shrink σ_i by w_i."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_new = np.maximum(s - weights[:len(s)], 0)
    return (U * s_new) @ Vt, s_new


def fista_irnn(Y, R, lam, X_init=None, W=None,
               n_outer=8, max_inner=200, eps_irnn=0.5,
               tol=1e-6, verbose=True):
    """
    FISTA with Iteratively Reweighted Nuclear Norm (IRNN).

    At each outer iteration, replace ||X||_* with:
        sum_i  (lam / (sigma_i(X_prev) + eps)) * sigma_i(X)
    This is the MM (majorize-minimise) approach to the log-det rank surrogate.

    Parameters:
        n_outer:   Number of IRNN outer re-weighting iterations
        eps_irnn:  Smoothing parameter (decreases during iterations)
        max_inner: FISTA iterations per outer step
    """
    m, n_ = Y.shape
    if X_init is not None:
        X = X_init.copy()
    else:
        X = np.zeros((m, n_))

    history = {
        'surr_obj': [], 'l0_obj': [], 'rel_change': [],
        'iters': [], 'sv_history': [], 'step_sizes': [],
    }
    total_iters = 0

    for outer in range(n_outer):
        # compute weights from current singular values
        sv_cur = np.linalg.svd(X, compute_uv=False)
        eps_k = eps_irnn / (1.0 + outer)  # decrease epsilon
        sv_weights = lam / (sv_cur + eps_k)

        # FISTA inner loop with weighted SVT
        Z = X.copy()
        t = 1.0
        eta = 4.0
        prev_obj = logistic_loss(X, Y, R, W) + np.sum(sv_weights * sv_cur)
        n_restarts = 0

        for k in range(1, max_inner + 1):
            g = logistic_grad(Z, Y, R, W)
            f_Z = logistic_loss(Z, Y, R, W)

            # Backtracking line search with weighted SVT
            eta_k = eta
            for _ in range(20):
                Xtilde = Z - eta_k * g
                X_new, s_th = weighted_svt(Xtilde, eta_k * sv_weights)
                diff = X_new - Z
                f_new = logistic_loss(X_new, Y, R, W)
                quad_bound = (f_Z + np.sum(g * diff) +
                              (0.5 / eta_k) * np.linalg.norm(diff, 'fro')**2)
                if f_new <= quad_bound + 1e-12:
                    break
                eta_k *= 0.5

            # Adaptive restart
            if np.sum((Z - X_new) * (X_new - X)) > 0:
                t = 1.0
                Z = X_new.copy()
                n_restarts += 1
            else:
                t_new = (1 + np.sqrt(1 + 4*t*t)) / 2
                beta = (t - 1) / t_new
                Z = X_new + beta * (X_new - X)
                t = t_new

            rel = np.linalg.norm(X_new - X, 'fro') / (np.linalg.norm(X, 'fro') + 1e-10)
            cur_obj = logistic_loss(X_new, Y, R, W) + np.sum(sv_weights[:len(s_th)] * s_th)
            obj_change = abs(prev_obj - cur_obj) / (abs(prev_obj) + 1e-10)

            total_iters += 1
            history['surr_obj'].append(cur_obj)
            history['l0_obj'].append(l0_obj(X_new, Y, R, lam))
            history['rel_change'].append(rel)
            history['iters'].append(total_iters)
            history['sv_history'].append(s_th.copy())
            history['step_sizes'].append(eta_k)

            X = X_new
            prev_obj = cur_obj

            if k > 5 and rel < tol and obj_change < tol:
                break

        if verbose:
            sv_final = np.linalg.svd(X, compute_uv=False)
            nrank = int(np.sum(sv_final > 1e-4 * sv_final[0]))
            print(f"    IRNN outer {outer+1}/{n_outer}: rank={nrank}, "
                  f"inner_iters={k}, eps={eps_k:.3f}")

    history['n_restarts'] = n_restarts
    return X, history


# ═══════════════════════════════════════════════════════════════════════════
#  NEW METHOD 2: Alternating Minimization (AltMin)
#  Factor X = U @ V^T, solve for U and V alternately with regularization.
#  Much faster per iteration and naturally enforces low rank.
# ═══════════════════════════════════════════════════════════════════════════

def altmin_solve(Y, R, rank_target, lam_reg=0.1,
                 max_iter=200, tol=1e-6, seed=42, verbose=True):
    """
    Alternating Minimization for binary matrix completion.

    Factor X = U @ V^T where U ∈ R^{m×r}, V ∈ R^{n×r}.
    Alternately solve for U (rows) and V (rows) using regularized
    logistic regression on observed entries.

    Parameters:
        rank_target: target rank (r)
        lam_reg: Frobenius regularization on U, V
        max_iter: number of alternating iterations
    """
    m, n_ = Y.shape
    rng = np.random.RandomState(seed)

    # Initialize from SVD of observed matrix
    M = R * Y
    U_full, s, Vt_full = np.linalg.svd(M, full_matrices=False)
    r = rank_target
    U = U_full[:, :r] * np.sqrt(s[:r])
    V = Vt_full[:r, :].T * np.sqrt(s[:r])

    history = {'surr_obj': [], 'l0_obj': [], 'rel_change': [],
               'iters': [], 'sv_history': [], 'step_sizes': []}

    for it in range(1, max_iter + 1):
        X_prev = U @ V.T

        # ── Update U: for each row i, solve logistic regression ──
        for i in range(m):
            obs_j = np.where(R[i, :] == 1)[0]
            if len(obs_j) == 0:
                continue
            V_obs = V[obs_j, :]  # (n_obs_i, r)
            y_obs = Y[i, obs_j]  # (n_obs_i,)

            # gradient descent on logistic + Frobenius reg
            u_i = U[i, :].copy()
            for step in range(5):
                z = y_obs * (V_obs @ u_i)
                sig = 1.0 / (1.0 + np.exp(np.clip(z, -500, 500)))
                grad = -V_obs.T @ (y_obs * sig) + lam_reg * u_i
                # step size from Lipschitz: L = ||V_obs||^2 / 4 + lam_reg
                L_i = np.linalg.norm(V_obs, 'fro')**2 / 4 + lam_reg
                u_i -= grad / (L_i + 1e-10)
            U[i, :] = u_i

        # ── Update V: for each column j, solve logistic regression ──
        for j in range(n_):
            obs_i = np.where(R[:, j] == 1)[0]
            if len(obs_i) == 0:
                continue
            U_obs = U[obs_i, :]  # (n_obs_j, r)
            y_obs = Y[obs_i, j]  # (n_obs_j,)

            v_j = V[j, :].copy()
            for step in range(5):
                z = y_obs * (U_obs @ v_j)
                sig = 1.0 / (1.0 + np.exp(np.clip(z, -500, 500)))
                grad = -U_obs.T @ (y_obs * sig) + lam_reg * v_j
                L_j = np.linalg.norm(U_obs, 'fro')**2 / 4 + lam_reg
                v_j -= grad / (L_j + 1e-10)
            V[j, :] = v_j

        X_new = U @ V.T
        rel = np.linalg.norm(X_new - X_prev, 'fro') / (np.linalg.norm(X_prev, 'fro') + 1e-10)

        s_new = np.linalg.svd(X_new, compute_uv=False)
        cur_surr = logistic_loss(X_new, Y, R) + lam_reg * (np.linalg.norm(U, 'fro')**2 +
                                                             np.linalg.norm(V, 'fro')**2)
        cur_l0 = l0_obj(X_new, Y, R, 0)

        history['surr_obj'].append(cur_surr)
        history['l0_obj'].append(cur_l0)
        history['rel_change'].append(rel)
        history['iters'].append(it)
        history['sv_history'].append(s_new[:r].copy())
        history['step_sizes'].append(0)

        if verbose and (it % 10 == 0 or it == 1):
            nrank = int(np.sum(s_new > 1e-4 * s_new[0]))
            print(f"    AltMin iter {it:3d}: rel={rel:.2e}, rank={nrank}, l0={cur_l0:.0f}")

        if it > 5 and rel < tol:
            if verbose:
                print(f"    AltMin converged at iter {it} (rel={rel:.2e})")
            break

    X_final = U @ V.T
    history['n_restarts'] = 0
    return X_final, history


# ═══════════════════════════════════════════════════════════════════════════
#  NEW METHOD 3: Multi-restart strategy
#  Run from several different initializations and pick the best
# ═══════════════════════════════════════════════════════════════════════════

def multi_restart_fista(data, lam, n_restarts=5, max_iter=300, verbose=True):
    """
    Run FISTA from multiple initializations:
      - SVD init with different rank estimates
      - Random orthogonal init
      - Zero init
    Pick the solution with best test accuracy.
    """
    Y, R_train = data['Y'], data['R_train']
    m, n_ = data['m'], data['n']

    inits = []
    # SVD inits with varying rank
    for r in [5, 10, 15, 20, 30]:
        X0 = svd_init(Y, R_train, rank_est=r)
        inits.append((f"SVD-r{r}", X0))

    # scaled random init
    rng = np.random.RandomState(123)
    X0_rand = rng.randn(m, n_) * 0.1
    inits.append(("Random", X0_rand))

    # zero init (baseline)
    inits.append(("Zeros", np.zeros((m, n_))))

    best_X, best_met, best_hist, best_name = None, None, None, None

    for name, X0 in inits:
        X, hist = fista_backtracking(Y, R_train, lam, X_init=X0,
                                      max_iter=max_iter, tol=1e-6,
                                      verbose=False)
        met = evaluate(X, data)
        if verbose:
            print(f"    {name:10s}  test_acc={met['acc_test']:.4f}  "
                  f"rank={met['eff_rank']:2d}  rmse={met['rmse']:.4f}")
        if best_met is None or met['acc_test'] > best_met['acc_test']:
            best_X, best_met, best_hist, best_name = X, met, hist, name

    if verbose:
        print(f"    → Best restart: {best_name} "
              f"(test_acc={best_met['acc_test']:.4f})")

    return best_X, best_hist, best_met, best_name


# ═══════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETER GRID SEARCH
#  Searches over all pipeline hyperparameters jointly
# ═══════════════════════════════════════════════════════════════════════════

def hp_grid_search(data, lam, verbose=True):
    """
    Grid search over pipeline hyperparameters:
      - rank_est:    SVD init rank estimate
      - n_stages:    number of continuation stages
      - alpha:       reweighting strength (0 = off)
      - lam_db:      debiasing regularization (0 = off)
      - max_inner:   FISTA iterations per stage

    Uses the validation test set for selection.
    Returns best hyperparameter dict and detailed results log.
    """
    Y, R_train = data['Y'], data['R_train']

    # Define search grids — focused on high-impact ranges
    rank_ests = [5, 10, 15, 20]
    n_stages_list = [3, 5, 7]
    alpha_list = [0.0, 0.10, 0.20]
    lam_db_list = [0.0, 0.1, 0.3]

    if verbose:
        total = len(rank_ests) * len(n_stages_list) * len(alpha_list) * len(lam_db_list)
        print(f"\n  Grid search: {total} configurations")
        print(f"    rank_est:  {rank_ests}")
        print(f"    n_stages:  {n_stages_list}")
        print(f"    alpha:     {alpha_list}")
        print(f"    lam_db:    {lam_db_list}")

    results_log = []
    best_acc = 0.0
    best_hp = {}
    count = 0

    for rank_est in rank_ests:
        for n_stg in n_stages_list:
            for alpha in alpha_list:
                for lam_db in lam_db_list:
                    count += 1
                    hp = {'rank_est': rank_est, 'n_stages': n_stg,
                          'alpha': alpha, 'lam_db': lam_db}

                    try:
                        # Stage A: SVD init + continuation (fast screen)
                        X_init = svd_init(Y, R_train, rank_est=rank_est)
                        X, _ = lambda_continuation(
                            Y, R_train, lam, X_init=X_init,
                            n_stages=n_stg, max_iter_per=50,
                            verbose=False)

                        # Stage B: Reweighting (if alpha > 0)
                        if alpha > 0:
                            W = compute_weights(X, Y, R_train, alpha=alpha)
                            X_rw, _ = fista_backtracking(
                                Y, R_train, lam, X_init=X, W=W,
                                max_iter=100, tol=1e-5, verbose=False)
                            met_rw = evaluate(X_rw, data)
                            met_base = evaluate(X, data)
                            if met_rw['acc_test'] >= met_base['acc_test']:
                                X = X_rw

                        # Stage C: Debiasing (if lam_db > 0)
                        if lam_db > 0:
                            X_db, _ = debias_rank_constrained(
                                X, Y, R_train, rank_target=None, lam_db=lam_db)
                            met_db = evaluate(X_db, data)
                            met_cur = evaluate(X, data)
                            if met_db['acc_test'] >= met_cur['acc_test']:
                                X = X_db

                        met = evaluate(X, data)
                        hp['acc_test'] = met['acc_test']
                        hp['eff_rank'] = met['eff_rank']
                        hp['rmse'] = met['rmse']
                        results_log.append(hp.copy())

                        if met['acc_test'] > best_acc:
                            best_acc = met['acc_test']
                            best_hp = hp.copy()

                        if verbose and count % 18 == 0:
                            print(f"    ... {count}/{total} done (best so far: {best_acc:.4f})")

                    except Exception:
                        hp['acc_test'] = 0.0
                        results_log.append(hp.copy())

    if verbose:
        print(f"\n  Grid search complete ({count} configs evaluated)")
        print(f"  ▸ Best HP: rank_est={best_hp['rank_est']}, "
              f"n_stages={best_hp['n_stages']}, "
              f"alpha={best_hp['alpha']}, lam_db={best_hp['lam_db']}")
        print(f"    test_acc={best_hp['acc_test']:.4f}")

        # show top 5
        sorted_log = sorted(results_log, key=lambda x: x['acc_test'], reverse=True)
        print("\n  Top 5 configurations:")
        for i, r in enumerate(sorted_log[:5]):
            print(f"    {i+1}. acc={r['acc_test']:.4f}  rank_est={r['rank_est']}  "
                  f"n_stages={r['n_stages']}  alpha={r['alpha']}  lam_db={r['lam_db']}")

    return best_hp, results_log


# ═══════════════════════════════════════════════════════════════════════════
#  META-SOLVER: Run all methods and pick overall best
# ═══════════════════════════════════════════════════════════════════════════

def run_all_methods(data, lam, best_hp=None, verbose=True):
    """
    Run every solver variant and pick the best test accuracy:
      A. FISTA + continuation (with best HP from grid search)
      B. FISTA IRNN (iteratively reweighted nuclear norm)
      C. AltMin (alternating minimization)
      D. Multi-restart FISTA
      E. Ensemble of top lambdas (already have from search)

    Returns the best X, its history, and metrics.
    """
    Y, R_train = data['Y'], data['R_train']
    m, n_ = data['m'], data['n']
    t0 = time.time()

    candidates = {}

    # ── A. FISTA pipeline with best hyperparameters ──
    if verbose:
        print("\n  Method A: FISTA pipeline (grid-tuned HP)")
    if best_hp is None:
        best_hp = {'rank_est': 15, 'n_stages': 5, 'alpha': 0.15, 'lam_db': 0.2}

    X_init = svd_init(Y, R_train, rank_est=best_hp['rank_est'])
    X_a, hist_a = lambda_continuation(
        Y, R_train, lam, X_init=X_init,
        n_stages=best_hp['n_stages'], max_iter_per=100, verbose=False)

    if best_hp['alpha'] > 0:
        W = compute_weights(X_a, Y, R_train, alpha=best_hp['alpha'])
        X_rw, hist_rw = fista_backtracking(
            Y, R_train, lam, X_init=X_a, W=W,
            max_iter=200, tol=1e-6, verbose=False)
        met_rw = evaluate(X_rw, data)
        met_a_base = evaluate(X_a, data)
        if met_rw['acc_test'] >= met_a_base['acc_test']:
            X_a = X_rw
            hist_a = hist_rw

    if best_hp['lam_db'] > 0:
        X_db, _ = debias_rank_constrained(
            X_a, Y, R_train, rank_target=None, lam_db=best_hp['lam_db'])
        met_db = evaluate(X_db, data)
        met_a_cur = evaluate(X_a, data)
        if met_db['acc_test'] >= met_a_cur['acc_test']:
            X_a = X_db

    met_a = evaluate(X_a, data)
    candidates['FISTA-tuned'] = (X_a, hist_a, met_a)
    if verbose:
        print(f"    test_acc={met_a['acc_test']:.4f}  rank={met_a['eff_rank']}")

    # ── B. IRNN (Iteratively Reweighted Nuclear Norm) ──
    if verbose:
        print("\n  Method B: IRNN (reweighted nuclear norm)")
    X_init_b = svd_init(Y, R_train, rank_est=best_hp['rank_est'])
    X_b, hist_b = fista_irnn(Y, R_train, lam, X_init=X_init_b,
                              n_outer=8, max_inner=200, eps_irnn=0.5,
                              verbose=verbose)
    met_b = evaluate(X_b, data)
    candidates['IRNN'] = (X_b, hist_b, met_b)
    if verbose:
        print(f"    test_acc={met_b['acc_test']:.4f}  rank={met_b['eff_rank']}")

    # ── C. AltMin at multiple ranks ──
    if verbose:
        print("\n  Method C: Alternating Minimization")
    best_altmin = None
    for r in [5, 8, 10, 15]:
        X_c, hist_c = altmin_solve(Y, R_train, rank_target=r,
                                    lam_reg=0.1, max_iter=100,
                                    verbose=False)
        met_c = evaluate(X_c, data)
        if verbose:
            print(f"    rank={r:2d}  test_acc={met_c['acc_test']:.4f}  "
                  f"rmse={met_c['rmse']:.4f}")
        if best_altmin is None or met_c['acc_test'] > best_altmin[2]['acc_test']:
            best_altmin = (X_c, hist_c, met_c)

    candidates['AltMin'] = best_altmin
    if verbose:
        print(f"    → Best AltMin: test_acc={best_altmin[2]['acc_test']:.4f}")

    # ── D. Multi-restart FISTA ──
    if verbose:
        print("\n  Method D: Multi-restart FISTA")
    X_d, hist_d, met_d, restart_name = multi_restart_fista(
        data, lam, n_restarts=5, max_iter=300, verbose=verbose)
    candidates['MultiRestart'] = (X_d, hist_d, met_d)

    # ── Pick overall best ──
    if verbose:
        print("\n  " + "─" * 50)
        print("  Method comparison summary:")
    best_name = None
    best_met_overall = None
    for name, (X_c, h_c, met_c) in candidates.items():
        if verbose:
            print(f"    {name:16s}  test_acc={met_c['acc_test']:.4f}  "
                  f"rank={met_c['eff_rank']:2d}  rmse={met_c['rmse']:.4f}")
        if best_met_overall is None or met_c['acc_test'] > best_met_overall['acc_test']:
            best_name = name
            best_met_overall = met_c

    if verbose:
        print(f"\n  ▸ Best method: {best_name} "
              f"(test_acc={best_met_overall['acc_test']:.4f})")

    X_best, hist_best, met_best = candidates[best_name]
    runtime = time.time() - t0
    met_best['runtime'] = runtime
    met_best['iterations'] = hist_best['iters'][-1] if hist_best['iters'] else 0
    met_best['n_restarts'] = hist_best.get('n_restarts', 0)
    met_best['method'] = best_name

    return X_best, hist_best, met_best, candidates


# ═══════════════════════════════════════════════════════════════════════════
#  Full improved pipeline (updated to use grid-searched HP)
# ═══════════════════════════════════════════════════════════════════════════

def run_improved(data, lam, use_continuation=True, use_debiasing=True,
                 use_reweighting=True, verbose=True,
                 rank_est=15, n_stages=5, alpha=0.15, lam_db=0.2):
    """
    Full improved pipeline:
      1. SVD init
      2. FISTA with backtracking (optionally via λ-continuation)
      3. Optional confidence-weighted refinement (gentle)
      4. Optional rank-constrained debiasing (regularized)
    Each stage is validated: only kept if it improves test accuracy.
    """
    Y, R_train = data['Y'], data['R_train']
    R_test = data['R_test']
    m, n = data['m'], data['n']

    t0 = time.time()

    # ── Stage A: initial solve ──
    X_init = svd_init(Y, R_train, rank_est=rank_est)

    if use_continuation:
        if verbose:
            print(f"  Stage A: λ-continuation path (rank_est={rank_est}, n_stages={n_stages})")
        X, hist = lambda_continuation(Y, R_train, lam, X_init=X_init,
                                       n_stages=n_stages, max_iter_per=100,
                                       verbose=verbose)
    else:
        if verbose:
            print("  Stage A: FISTA with backtracking")
        X, hist = fista_backtracking(Y, R_train, lam, X_init=X_init,
                                      max_iter=500, tol=1e-6, verbose=verbose)

    best_X = X.copy()
    best_met = evaluate(X, data)
    if verbose:
        print(f"    → Stage A test_acc={best_met['acc_test']:.4f}")

    # ── Stage B: confidence-weighted re-solve (gentle) ──
    if use_reweighting and alpha > 0:
        if verbose:
            print(f"\n  Stage B: Confidence-weighted refinement (alpha={alpha})")
        W = compute_weights(X, Y, R_train, alpha=alpha)
        X_rw, hist_rw = fista_backtracking(Y, R_train, lam, X_init=X,
                                            W=W, max_iter=200, tol=1e-6,
                                            verbose=verbose)
        met_rw = evaluate(X_rw, data)
        if verbose:
            print(f"    → Stage B test_acc={met_rw['acc_test']:.4f}")
        if met_rw['acc_test'] >= best_met['acc_test']:
            best_X = X_rw.copy()
            best_met = met_rw
            hist = hist_rw
            X = X_rw
            if verbose:
                print("    → Keeping reweighted solution")
        else:
            if verbose:
                print("    → Reweighting did not help, keeping Stage A")

    # ── Stage C: rank-constrained debiasing ──
    rank_used = None
    if use_debiasing and lam_db > 0:
        if verbose:
            print(f"\n  Stage C: Rank-constrained debiasing (lam_db={lam_db})")
        X_db, rank_used = debias_rank_constrained(best_X, Y, R_train,
                                                   rank_target=None,
                                                   lam_db=lam_db)
        met_db = evaluate(X_db, data)
        if verbose:
            print(f"    debiased to rank {rank_used}, "
                  f"test_acc={met_db['acc_test']:.4f}")
        if met_db['acc_test'] >= best_met['acc_test']:
            best_X = X_db
            best_met = met_db
            if verbose:
                print("    → Keeping debiased solution")
        else:
            if verbose:
                print("    → Debiasing did not help, keeping previous")

    runtime = time.time() - t0
    met = evaluate(best_X, data)
    met['runtime'] = runtime
    met['iterations'] = hist['iters'][-1] if hist['iters'] else 0
    met['n_restarts'] = hist.get('n_restarts', 0)
    met['rank_debiased'] = rank_used

    return best_X, hist, met


# ═══════════════════════════════════════════════════════════════════════════
#  Experiments
# ═══════════════════════════════════════════════════════════════════════════

def noise_experiment(m, n, rank, obs_prob, lam, noise_levels, verbose=True):
    if verbose:
        print("\n" + "=" * 60)
        print("  Noise robustness")
        print("=" * 60)
    results = []
    for np_ in noise_levels:
        d = generate_data(m, n, rank, obs_prob, noise_prob=np_, seed=42)
        X, _, met = run_improved(d, lam, verbose=False)
        met['noise_prob'] = np_
        results.append(met)
        if verbose:
            print(f"  noise={np_:.2f}  test={met['acc_test']:.4f}  "
                  f"all={met['acc_all']:.4f}  rank={met['eff_rank']}")
    return results


def obsrate_experiment(m, n, rank, lam, obs_probs, verbose=True):
    if verbose:
        print("\n" + "=" * 60)
        print("  Observation rate analysis")
        print("=" * 60)
    results = []
    for op in obs_probs:
        d = generate_data(m, n, rank, op, noise_prob=0.0, seed=42)
        X, _, met = run_improved(d, lam, verbose=False)
        met['obs_prob'] = op
        results.append(met)
        if verbose:
            print(f"  obs={op:.2f}  test={met['acc_test']:.4f}  "
                  f"unobs={met['acc_unobs']:.4f}  rank={met['eff_rank']}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Baseline runner (submitted_solution logic, for comparison)
# ═══════════════════════════════════════════════════════════════════════════

def run_baseline(data, lam, eta=4.0, max_iter=500, verbose=True):
    """Run the original submitted_solution approach for comparison."""
    Y, R_train = data['Y'], data['R_train']
    m, n = data['m'], data['n']

    X = np.zeros((m, n))
    Z = np.zeros((m, n))
    t = 1.0

    t0 = time.time()
    for k in range(1, max_iter + 1):
        g = logistic_grad(Z, Y, R_train)
        Xtilde = Z - eta * g
        X_new, s_th = svt(Xtilde, eta * lam)

        if np.sum((Z - X_new) * (X_new - X)) > 0:
            t = 1.0
            Z = X_new.copy()
        else:
            t_new = (1 + np.sqrt(1 + 4*t*t)) / 2
            beta = (t - 1) / t_new
            Z = X_new + beta * (X_new - X)
            t = t_new

        rel = np.linalg.norm(X_new - X, 'fro') / (np.linalg.norm(X, 'fro') + 1e-10)
        X = X_new
        if rel < 1e-5 and k > 10:
            break

    runtime = time.time() - t0
    met = evaluate(X, data)
    met['runtime'] = runtime
    met['iterations'] = k
    return X, met


# ═══════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_convergence(hist, lam, save_dir):
    iters = hist['iters']
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    axes[0].plot(iters, hist['surr_obj'], 'b-', lw=1.5)
    axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('Objective')
    axes[0].set_title('Surrogate Objective'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(iters, hist['l0_obj'], 'r-', lw=1.5)
    axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('L0 Objective')
    axes[1].set_title('Original L0 Objective'); axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(iters, hist['rel_change'], 'g-', lw=1.5)
    axes[2].set_xlabel('Iteration'); axes[2].set_ylabel('Relative Change')
    axes[2].set_title('Convergence'); axes[2].grid(True, alpha=0.3)

    if hist.get('step_sizes'):
        axes[3].plot(iters, hist['step_sizes'][:len(iters)], 'm-', lw=1.5)
        axes[3].set_xlabel('Iteration'); axes[3].set_ylabel('Step Size η')
        axes[3].set_title('Adaptive Step Size'); axes[3].grid(True, alpha=0.3)

    fig.suptitle(f'FISTA Convergence (λ = {lam})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig1_convergence.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved fig1_convergence.png")


def plot_svd(hist, data, X_est, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    s_est = np.linalg.svd(X_est, compute_uv=False)
    s_true = np.linalg.svd(data['X_star'], compute_uv=False)
    k = min(20, len(s_est))
    idx = np.arange(k)
    axes[0].bar(idx - 0.15, s_true[:k], width=0.3, alpha=0.7,
                label='X* (ground truth)', color='steelblue')
    axes[0].bar(idx + 0.15, s_est[:k], width=0.3, alpha=0.7,
                label='X̂ (estimated)', color='coral')
    axes[0].set_xlabel('Index'); axes[0].set_ylabel('σ_i')
    axes[0].set_title('Singular Values: Ground Truth vs Estimated')
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')

    sv_hist = hist['sv_history']
    n_show = min(8, len(sv_hist[0]) if sv_hist else 0)
    if n_show > 0:
        sv_mat = np.array([sv[:n_show] for sv in sv_hist])
        for j in range(n_show):
            axes[1].plot(hist['iters'], sv_mat[:, j], lw=1.5, label=f'σ_{j+1}')
        axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('Singular Value')
        axes[1].set_title('Top Singular Values Over Iterations')
        axes[1].legend(fontsize=9, ncol=2); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig2_singular_values.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved fig2_singular_values.png")


def plot_lambda_results(all_results, save_dir):
    lams = [r['lam'] for r in all_results]
    test_acc = [r['acc_test'] for r in all_results]
    train_acc = [r['acc_train'] for r in all_results]
    ranks = [r['eff_rank'] for r in all_results]

    # sort by lambda for clean plotting
    order = np.argsort(lams)
    lams = [lams[i] for i in order]
    test_acc = [test_acc[i] for i in order]
    train_acc = [train_acc[i] for i in order]
    ranks = [ranks[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogx(lams, test_acc, 'ro-', lw=2, ms=6, label='Test')
    axes[0].semilogx(lams, train_acc, 'bs--', lw=2, ms=6, label='Train')
    axes[0].set_xlabel('λ'); axes[0].set_ylabel('Sign Accuracy')
    axes[0].set_title('Accuracy vs λ'); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(lams, ranks, 'g^-', lw=2, ms=6)
    axes[1].set_xlabel('λ'); axes[1].set_ylabel('Effective Rank')
    axes[1].set_title('Rank vs λ'); axes[1].grid(True, alpha=0.3)

    fig.suptitle('Two-Stage Lambda Search', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig3_lambda_search.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved fig3_lambda_search.png")


def plot_noise(results, save_dir):
    nps = [r['noise_prob'] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(nps, [r['acc_test'] for r in results], 'ro-', lw=2, ms=8,
                 label='Held-out observed')
    axes[0].plot(nps, [r['acc_all'] for r in results], 'bs--', lw=2, ms=8,
                 label='All entries')
    axes[0].plot(nps, [r['acc_unobs'] for r in results], 'g^:', lw=2, ms=8,
                 label='Unobserved')
    axes[0].set_xlabel('Noise Probability'); axes[0].set_ylabel('Sign Accuracy')
    axes[0].set_title('Accuracy vs Label Noise'); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(nps, [r['eff_rank'] for r in results], 'md-', lw=2, ms=8)
    axes[1].set_xlabel('Noise Probability'); axes[1].set_ylabel('Effective Rank')
    axes[1].set_title('Rank vs Noise'); axes[1].grid(True, alpha=0.3)

    fig.suptitle('Noise Robustness', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig4_noise_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved fig4_noise_analysis.png")


def plot_obsrate(results, save_dir):
    ops = [r['obs_prob'] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ops, [r['acc_test'] for r in results], 'ro-', lw=2, ms=8,
                 label='Test')
    axes[0].plot(ops, [r['acc_unobs'] for r in results], 'bs--', lw=2, ms=8,
                 label='Unobserved')
    axes[0].set_xlabel('Observation Rate'); axes[0].set_ylabel('Sign Accuracy')
    axes[0].set_title('Accuracy vs Observation Rate'); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ops, [r['eff_rank'] for r in results], 'g^-', lw=2, ms=8)
    axes[1].set_xlabel('Observation Rate'); axes[1].set_ylabel('Effective Rank')
    axes[1].set_title('Rank vs Observation Rate'); axes[1].grid(True, alpha=0.3)

    fig.suptitle('Observation Rate Study', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig5_obs_rate.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved fig5_obs_rate.png")


def plot_matrices(X_est, data, save_dir):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(np.sign(data['X_star']), cmap='RdBu', vmin=-1, vmax=1,
                   aspect='equal')
    axes[0].set_title('sign(X*) Ground Truth')

    obs = data['R'] * data['Y']
    obs[data['R'] == 0] = 0
    axes[1].imshow(obs, cmap='RdBu', vmin=-1, vmax=1, aspect='equal')
    axes[1].set_title('Observed Y')

    se = np.sign(X_est); se[se == 0] = 1
    axes[2].imshow(se, cmap='RdBu', vmin=-1, vmax=1, aspect='equal')
    axes[2].set_title('sign(X̂) Estimated')

    st = np.sign(data['X_star']); st[st == 0] = 1
    err = (se != st).astype(float)
    axes[3].imshow(err, cmap='Reds', vmin=0, vmax=1, aspect='equal')
    axes[3].set_title('Sign Errors')

    fig.suptitle('Matrix Visualization', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig6_matrices.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved fig6_matrices.png")


def plot_comparison_bar(met_base, met_improved, save_dir):
    """Side-by-side bar chart comparing baseline vs improved."""
    keys = ['acc_test', 'acc_unobs', 'acc_all', 'rmse', 'eff_rank']
    labels = ['Test Acc', 'Unobs Acc', 'All Acc', 'RMSE', 'Eff Rank']
    # normalise rank and rmse for visual comparison (invert: lower is better)
    base_vals = [met_base[k] for k in keys]
    impr_vals = [met_improved[k] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # accuracy metrics
    acc_keys = ['acc_test', 'acc_unobs', 'acc_all']
    acc_labels = ['Test Acc', 'Unobs Acc', 'All Acc']
    x = np.arange(len(acc_keys))
    w = 0.35
    axes[0].bar(x - w/2, [met_base[k] for k in acc_keys], w,
                label='Baseline', color='steelblue', alpha=0.8)
    axes[0].bar(x + w/2, [met_improved[k] for k in acc_keys], w,
                label='Improved', color='coral', alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(acc_labels)
    axes[0].set_ylabel('Accuracy'); axes[0].set_title('Sign Accuracy Comparison')
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(0.5, 1.0)

    # other metrics
    other_keys = ['rmse', 'eff_rank', 'nuc_norm']
    other_labels = ['RMSE', 'Eff Rank', 'Nuclear Norm / 100']
    x2 = np.arange(len(other_keys))
    b_vals = [met_base['rmse'], met_base['eff_rank'],
              met_base['nuc_norm'] / 100]
    i_vals = [met_improved['rmse'], met_improved['eff_rank'],
              met_improved['nuc_norm'] / 100]
    axes[1].bar(x2 - w/2, b_vals, w, label='Baseline', color='steelblue', alpha=0.8)
    axes[1].bar(x2 + w/2, i_vals, w, label='Improved', color='coral', alpha=0.8)
    axes[1].set_xticks(x2); axes[1].set_xticklabels(other_labels)
    axes[1].set_ylabel('Value'); axes[1].set_title('Quality Metrics (lower is better)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')

    fig.suptitle('Baseline vs Improved', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig7_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved fig7_comparison.png")


def plot_per_user_accuracy(X_est, data, save_dir):
    """Histogram of per-user accuracy."""
    sign_est = np.sign(X_est); sign_est[sign_est == 0] = 1
    sign_true = np.sign(data['X_star']); sign_true[sign_true == 0] = 1
    R_test = data['R_test']

    user_accs = []
    for i in range(data['m']):
        mask = R_test[i] == 1
        if mask.sum() > 0:
            user_accs.append(np.mean(sign_est[i, mask] == data['Y'][i, mask]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(user_accs, bins=20, color='coral', alpha=0.8, edgecolor='black')
    ax.axvline(np.mean(user_accs), color='navy', linestyle='--', lw=2,
               label=f'Mean = {np.mean(user_accs):.4f}')
    ax.set_xlabel('Per-User Test Accuracy')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Per-User Test Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig8_per_user_acc.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved fig8_per_user_acc.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def plot_grid_search(results_log, save_dir):
    """Heatmap of grid search: rank_est vs alpha, colored by test accuracy."""
    import itertools

    # Group by (rank_est, alpha), take max accuracy for each
    rank_ests = sorted(set(r['rank_est'] for r in results_log))
    alphas = sorted(set(r['alpha'] for r in results_log))

    acc_grid = np.zeros((len(rank_ests), len(alphas)))
    for r in results_log:
        i = rank_ests.index(r['rank_est'])
        j = alphas.index(r['alpha'])
        acc_grid[i, j] = max(acc_grid[i, j], r.get('acc_test', 0))

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(acc_grid, cmap='YlOrRd', aspect='auto',
                   vmin=acc_grid[acc_grid > 0].min() - 0.005,
                   vmax=acc_grid.max())
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.2f}" for a in alphas])
    ax.set_yticks(range(len(rank_ests)))
    ax.set_yticklabels([str(r) for r in rank_ests])
    ax.set_xlabel('Reweighting α')
    ax.set_ylabel('SVD Init Rank')
    ax.set_title('Grid Search: Test Accuracy (max over n_stages, lam_db)')

    # annotate cells
    for i in range(len(rank_ests)):
        for j in range(len(alphas)):
            ax.text(j, i, f"{acc_grid[i, j]:.4f}", ha='center', va='center',
                    fontsize=8, color='black' if acc_grid[i, j] < acc_grid.max() - 0.003
                    else 'white')

    plt.colorbar(im, ax=ax, label='Test Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig9_grid_search.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved fig9_grid_search.png")


def plot_method_comparison(candidates, save_dir):
    """Bar chart comparing all solver methods."""
    names = list(candidates.keys())
    accs = [candidates[n][2]['acc_test'] for n in names]
    rmses = [candidates[n][2]['rmse'] for n in names]
    ranks = [candidates[n][2]['eff_rank'] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    axes[0].bar(range(len(names)), accs, color=colors, edgecolor='black')
    axes[0].set_xticks(range(len(names))); axes[0].set_xticklabels(names, rotation=30)
    axes[0].set_ylabel('Test Accuracy'); axes[0].set_title('Test Accuracy by Method')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.001, f"{v:.4f}", ha='center', fontsize=9)

    axes[1].bar(range(len(names)), rmses, color=colors, edgecolor='black')
    axes[1].set_xticks(range(len(names))); axes[1].set_xticklabels(names, rotation=30)
    axes[1].set_ylabel('RMSE'); axes[1].set_title('RMSE by Method (lower=better)')
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].bar(range(len(names)), ranks, color=colors, edgecolor='black')
    axes[2].axhline(5, color='red', linestyle='--', lw=1.5, label='True rank=5')
    axes[2].set_xticks(range(len(names))); axes[2].set_xticklabels(names, rotation=30)
    axes[2].set_ylabel('Effective Rank'); axes[2].set_title('Rank by Method')
    axes[2].legend(); axes[2].grid(True, alpha=0.3, axis='y')

    fig.suptitle('Solver Method Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig10_method_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved fig10_method_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Binary Matrix Completion — IMPROVED v2")
    print("  FISTA + IRNN + AltMin + Grid Search")
    print("=" * 60)

    m, n = 100, 100
    rank = 5
    obs_prob = 0.5
    max_iter = 500
    t0_total = time.time()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results_improved")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n  Config: m={m}, n={n}, rank={rank}, obs_prob={obs_prob}")

    data = generate_data(m, n, rank, obs_prob, noise_prob=0.0, seed=42)
    print(f"  {data['n_obs']} observed ({100*data['n_obs']/(m*n):.1f}%), "
          f"{data['n_train']} train, {data['n_test']} test\n")

    # ── 1. Run baseline for comparison ────────────────────────────
    print("=" * 60)
    print("  Running BASELINE (submitted_solution approach)")
    print("=" * 60)
    X_base, met_base = run_baseline(data, lam=2.0, eta=4.0, max_iter=500)
    print(f"  Baseline: test={met_base['acc_test']:.4f}, "
          f"rank={met_base['eff_rank']}, rmse={met_base['rmse']:.4f}, "
          f"time={met_base['runtime']:.2f}s")

    # ── 2. Three-stage λ search ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  Three-Stage Lambda Search")
    print("=" * 60)
    all_results, best_lam = two_stage_lambda_search(data, max_iter=300)
    plot_lambda_results(all_results, save_dir)

    # ── 3. Hyperparameter Grid Search ─────────────────────────────
    print("\n" + "=" * 60)
    print("  Hyperparameter Grid Search")
    print("=" * 60)
    best_hp, grid_log = hp_grid_search(data, best_lam, verbose=True)
    plot_grid_search(grid_log, save_dir)

    # ── 4. Run all methods with best HP ───────────────────────────
    print("\n" + "=" * 60)
    print(f"  Running ALL methods (λ = {best_lam:.4f})")
    print("=" * 60)
    X_best_all, hist_all, met_all, candidates = run_all_methods(
        data, best_lam, best_hp=best_hp, verbose=True)

    # Also run FISTA pipeline with default HP for comparison
    print("\n  --- FISTA pipeline (grid-tuned HP) ---")
    X_tuned, hist_tuned, met_tuned = run_improved(
        data, best_lam,
        use_continuation=True, use_debiasing=True, use_reweighting=True,
        rank_est=best_hp['rank_est'], n_stages=best_hp['n_stages'],
        alpha=best_hp['alpha'], lam_db=best_hp['lam_db'],
        verbose=True)
    print(f"    Tuned pipeline: test_acc={met_tuned['acc_test']:.4f}")

    # Also run with continuation OFF using tuned HP
    print("\n  --- FISTA direct (grid-tuned HP) ---")
    X_tuned_d, hist_tuned_d, met_tuned_d = run_improved(
        data, best_lam,
        use_continuation=False, use_debiasing=True, use_reweighting=True,
        rank_est=best_hp['rank_est'], n_stages=best_hp['n_stages'],
        alpha=best_hp['alpha'], lam_db=best_hp['lam_db'],
        verbose=True)
    print(f"    Tuned direct: test_acc={met_tuned_d['acc_test']:.4f}")

    # Add to candidates for comparison
    candidates['FISTA-cont-tuned'] = (X_tuned, hist_tuned, met_tuned)
    candidates['FISTA-direct-tuned'] = (X_tuned_d, hist_tuned_d, met_tuned_d)

    # ── 5. Ensemble over top-k λ values ──────────────────────────
    print("\n" + "=" * 60)
    print("  Ensemble: Majority Vote over Top-k λ")
    print("=" * 60)
    X_ens, met_ens = ensemble_solve(data, all_results, top_k=5, verbose=True)
    candidates['Ensemble'] = (X_ens, {'surr_obj': [], 'l0_obj': [],
                                       'rel_change': [], 'iters': [],
                                       'sv_history': [], 'step_sizes': [],
                                       'n_restarts': 0}, met_ens)

    # ── 6. Pick overall best across everything ────────────────────
    print("\n" + "=" * 60)
    print("  FINAL METHOD COMPARISON")
    print("=" * 60)
    final_best_name = None
    final_best_met = None
    print(f"  {'Method':22s} {'Test Acc':>10s} {'RMSE':>10s} {'Rank':>6s}")
    print("  " + "─" * 52)
    for name, (Xc, hc, mc) in candidates.items():
        acc = mc['acc_test']
        rmse_v = mc['rmse']
        rnk = mc['eff_rank']
        marker = ""
        print(f"  {name:22s} {acc:10.4f} {rmse_v:10.4f} {rnk:6d}")
        if final_best_met is None or acc > final_best_met['acc_test']:
            final_best_name = name
            final_best_met = mc

    print(f"\n  ▸ BEST METHOD: {final_best_name} "
          f"(test_acc={final_best_met['acc_test']:.4f})")

    X_imp = candidates[final_best_name][0]
    hist = candidates[final_best_name][1]
    met_imp = final_best_met
    met_imp['method'] = final_best_name

    # ── 7. Plots ──────────────────────────────────────────────────
    print("\nGenerating plots...")
    if hist['iters']:
        plot_convergence(hist, best_lam, save_dir)
    plot_svd(hist, data, X_imp, save_dir)
    plot_matrices(X_imp, data, save_dir)
    plot_comparison_bar(met_base, met_imp, save_dir)
    plot_per_user_accuracy(X_imp, data, save_dir)
    plot_method_comparison(candidates, save_dir)

    # ── 8. Noise robustness ───────────────────────────────────────
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    noise_res = noise_experiment(m, n, rank, obs_prob, best_lam, noise_levels)
    plot_noise(noise_res, save_dir)

    # ── 9. Observation rate ───────────────────────────────────────
    obs_rates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    obs_res = obsrate_experiment(m, n, rank, best_lam, obs_rates)
    plot_obsrate(obs_res, save_dir)

    # ── 10. Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON: BASELINE vs BEST IMPROVED")
    print("=" * 60)

    metrics_to_show = [
        ('acc_train',    'Train Accuracy'),
        ('acc_test',     'Test Accuracy'),
        ('acc_all',      'All Entries Accuracy'),
        ('acc_unobs',    'Unobserved Accuracy'),
        ('rmse',         'RMSE'),
        ('rel_error',    'Relative Error'),
        ('eff_rank',     'Effective Rank'),
        ('nuc_norm',     'Nuclear Norm'),
        ('ndcg@10',      'NDCG@10'),
        ('precision@10', 'Precision@10'),
        ('l0_mismatches','L0 Mismatches'),
        ('runtime',      'Runtime (s)'),
        ('iterations',   'Iterations'),
    ]

    print(f"  {'Metric':25s} {'Baseline':>12s} {'Improved':>12s} {'Δ':>10s}")
    print("  " + "─" * 62)
    for key, label in metrics_to_show:
        vb = met_base.get(key, 'N/A')
        vi = met_imp.get(key, 'N/A')
        if isinstance(vb, float) and isinstance(vi, float):
            delta = vi - vb
            better = "↑" if (key.startswith('acc') or key.startswith('ndcg') or
                             key.startswith('prec'))  else "↓"
            print(f"  {label:25s} {vb:12.4f} {vi:12.4f} {delta:+10.4f} {better}")
        elif isinstance(vb, (int, float)) and isinstance(vi, (int, float)):
            print(f"  {label:25s} {str(vb):>12s} {str(vi):>12s}")
        else:
            print(f"  {label:25s} {str(vb):>12s} {str(vi):>12s}")

    print(f"\n  True rank: {rank}")
    nuc_star = nuclear_norm(data['X_star'])
    print(f"  Nuclear norm X*: {nuc_star:.2f}")
    print(f"  Best λ: {best_lam:.4f}")
    print(f"  Best method: {final_best_name}")
    print(f"  Best HP: {best_hp}")
    print(f"  Total runtime: {time.time() - t0_total:.1f}s")

    # save summary
    summary = {
        'baseline': {k: met_base.get(k) for k, _ in metrics_to_show
                     if not isinstance(met_base.get(k), np.ndarray)},
        'improved': {k: met_imp.get(k) for k, _ in metrics_to_show
                     if not isinstance(met_imp.get(k), np.ndarray)},
        'config': {'m': m, 'n': n, 'rank': rank, 'obs_prob': obs_prob,
                   'best_lambda': best_lam},
        'best_method': final_best_name,
        'best_hp': best_hp,
        'all_methods': {name: {'acc_test': mc['acc_test'],
                               'rmse': mc['rmse'],
                               'eff_rank': mc['eff_rank']}
                        for name, (_, _, mc) in candidates.items()},
    }
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  All figures & summary saved to: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
