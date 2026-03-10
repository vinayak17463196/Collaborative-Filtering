"""
Midsem Exam - Binary Matrix Completion
Vinayak Agrawal (2022574)

We want to solve:  min_X  ||Y - R*X||_0 + lambda * ||X||_*
But L0 is non-differentiable, so we relax it with logistic loss and
use FISTA + SVT to solve the resulting convex problem.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os


# ---- Data generation ----

def generate_data(m=100, n=100, rank=5, obs_prob=0.5,
                  noise_prob=0.0, test_frac=0.2, seed=42):
    """
    Create synthetic low-rank matrix, observation mask, and binary labels.
    Split observed entries into train/test sets.
    """
    rng = np.random.RandomState(seed)

    # low-rank ground truth: X* = A @ B^T
    A = rng.randn(m, rank) / np.sqrt(rank)
    B = rng.randn(n, rank) / np.sqrt(rank)
    X_star = A @ B.T

    # observation mask
    R = (rng.rand(m, n) < obs_prob).astype(np.float64)

    # binary labels from sign of X*
    Y_clean = np.sign(X_star)
    Y_clean[Y_clean == 0] = 1.0  # edge case

    # optionally flip some labels
    Y = Y_clean.copy()
    if noise_prob > 0:
        flip = rng.rand(m, n) < noise_prob
        Y[flip] *= -1.0

    # train/test split among observed entries
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


# ---- Loss and gradient ----

def logistic_loss(X, Y, R):
    """
    Logistic loss over observed entries only:
    sum_{R_ij=1} log(1 + exp(-Y_ij * X_ij))
    """
    # using logaddexp for numerical stability
    return np.sum(R * np.logaddexp(0, -Y * X))


def logistic_grad(X, Y, R):
    """
    Gradient: nabla f(X)_ij = -R_ij * Y_ij * sigmoid(-Y_ij * X_ij)
    """
    z = Y * X
    sig = 1.0 / (1.0 + np.exp(np.clip(z, -500, 500)))
    return -R * Y * sig


# ---- Nuclear norm stuff ----

def nuclear_norm(X):
    """Sum of singular values."""
    return np.sum(np.linalg.svd(X, compute_uv=False))


def svt(X, tau):
    """
    Singular Value Thresholding (proximal operator for nuclear norm).
    Returns thresholded matrix and the thresholded singular values.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_new = np.maximum(s - tau, 0)
    return (U * s_new) @ Vt, s_new


# ---- Objectives (for tracking) ----

def compute_surrogate_obj(X, Y, R, lam):
    """Logistic loss + lambda * nuclear norm"""
    return logistic_loss(X, Y, R) + lam * nuclear_norm(X)


def compute_l0_obj(X, Y, R, lam):
    """
    Original L0 objective for monitoring purposes.
    Counts sign mismatches on observed entries + lambda*||X||_*
    """
    sx = np.sign(X)
    sx[sx == 0] = 1.0
    mismatch = np.sum(R * (sx != Y))
    return mismatch + lam * nuclear_norm(X)


# ---- FISTA solver with adaptive restart ----

def fista(Y, R, lam, eta=4.0, max_iter=500, tol=1e-5, verbose=True):
    """
    Accelerated FISTA with optimal step size and adaptive restart.
    
    Key ideas:
      - eta = 1/L = 4.0 is the theoretically optimal step size since
        the Lipschitz constant of the logistic gradient is L = 1/4.
      - Adaptive restart (O'Donoghue & Candes, 2015): when the Nesterov
        momentum overshoots (gradient restart condition), we reset t=1
        to avoid oscillation and ensure monotonic convergence.
    
    Each iteration:
      1) Gradient step: Xtilde = Z - eta * grad_f(Z)
      2) Proximal step: X_new = SVT(Xtilde, eta*lambda)
      3) Restart check: if <Z - X_new, X_new - X> > 0, reset momentum
      4) Otherwise: standard Nesterov extrapolation for next Z
    
    Returns the estimated matrix and a dict with history.
    """
    m, n = Y.shape
    X = np.zeros((m, n))
    Z = np.zeros((m, n))  # extrapolated point
    t = 1.0
    n_restarts = 0

    history = {
        'surr_obj': [], 'l0_obj': [], 'rel_change': [],
        'iters': [], 'sv_history': []
    }

    for k in range(1, max_iter + 1):
        # gradient step on the extrapolated point Z
        g = logistic_grad(Z, Y, R)
        Xtilde = Z - eta * g

        # proximal step (SVT)
        X_new, s_th = svt(Xtilde, eta * lam)

        # adaptive restart check (O'Donoghue & Candes, 2015):
        # if momentum is pushing in a bad direction, reset
        if np.sum((Z - X_new) * (X_new - X)) > 0:
            t = 1.0
            Z = X_new.copy()
            n_restarts += 1
        else:
            # standard Nesterov momentum update
            t_new = (1 + np.sqrt(1 + 4*t*t)) / 2
            beta = (t - 1) / t_new
            Z = X_new + beta * (X_new - X)
            t = t_new

        # check convergence
        diff = np.linalg.norm(X_new - X, 'fro')
        rel = diff / (np.linalg.norm(X, 'fro') + 1e-10)

        # record stuff
        obj_s = compute_surrogate_obj(X_new, Y, R, lam)
        obj_l0 = compute_l0_obj(X_new, Y, R, lam)
        history['surr_obj'].append(obj_s)
        history['l0_obj'].append(obj_l0)
        history['rel_change'].append(rel)
        history['iters'].append(k)
        history['sv_history'].append(s_th.copy())

        if verbose and (k % 10 == 0 or k == 1):
            nrank = np.sum(s_th > 1e-8)
            print(f"  iter {k:4d} | surr={obj_s:.1f} | l0={obj_l0:.1f} | "
                  f"rel_chg={rel:.2e} | rank={nrank}")

        X = X_new

        if rel < tol and k > 10:
            if verbose:
                print(f"  converged at iter {k} (rel_change={rel:.2e}, "
                      f"restarts={n_restarts})")
            break

    history['n_restarts'] = n_restarts
    return X, history


# ---- Evaluation ----

def evaluate(X_est, data):
    """Compute accuracy metrics and other stats."""
    X_star = data['X_star']
    Y = data['Y']
    R_train, R_test, R = data['R_train'], data['R_test'], data['R']

    sign_est = np.sign(X_est)
    sign_est[sign_est == 0] = 1.0
    sign_true = np.sign(X_star)
    sign_true[sign_true == 0] = 1.0

    # accuracy on different subsets
    tr_mask = R_train == 1
    te_mask = R_test == 1
    unobs = R == 0

    acc_train = np.mean(sign_est[tr_mask] == Y[tr_mask])
    acc_test = np.mean(sign_est[te_mask] == Y[te_mask])
    acc_all = np.mean(sign_est == sign_true)
    acc_unobs = np.mean(sign_est[unobs] == sign_true[unobs])

    rmse = np.sqrt(np.mean((X_est - X_star)**2))
    s = np.linalg.svd(X_est, compute_uv=False)
    eff_rank = int(np.sum(s > 1e-4 * s[0]))

    return {
        'acc_train': acc_train, 'acc_test': acc_test,
        'acc_all': acc_all, 'acc_unobs': acc_unobs,
        'rmse': rmse, 'eff_rank': eff_rank,
        'nuc_norm': np.sum(s), 'svs': s,
        'l0_mismatches': int(np.sum(R_train * (sign_est != Y)))
    }


# ---- Experiments ----

def run_convergence(data, lam, eta=4.0, max_iter=500):
    """Run FISTA with given lambda and report results."""
    print("=" * 60)
    print(f"Convergence experiment (lam={lam})")
    print("=" * 60)

    t0 = time.time()
    X_est, hist = fista(data['Y'], data['R_train'], lam,
                        eta=eta, max_iter=max_iter)
    runtime = time.time() - t0

    met = evaluate(X_est, data)
    print(f"\n  runtime:        {runtime:.2f}s")
    print(f"  iterations:     {hist['iters'][-1]}")
    print(f"  restarts:       {hist.get('n_restarts', 0)}")
    print(f"  train acc:      {met['acc_train']:.4f}")
    print(f"  test acc:       {met['acc_test']:.4f}")
    print(f"  all acc:        {met['acc_all']:.4f}")
    print(f"  unobs acc:      {met['acc_unobs']:.4f}")
    print(f"  rmse:           {met['rmse']:.4f}")
    print(f"  eff rank:       {met['eff_rank']}")
    print(f"  nuc norm:       {met['nuc_norm']:.2f}")
    print(f"  mismatches:     {met['l0_mismatches']}")

    return X_est, hist, met, runtime


def lambda_search(data, lambdas, eta=4.0, max_iter=300):
    """Try different lambda values and pick the best one based on test acc."""
    print("\n" + "=" * 60)
    print("Lambda grid search")
    print("=" * 60)

    results = []
    for lam in lambdas:
        print(f"  lam={lam:.4f} ... ", end="", flush=True)
        X_est, _ = fista(data['Y'], data['R_train'], lam,
                         eta=eta, max_iter=max_iter, verbose=False)
        met = evaluate(X_est, data)
        met['lam'] = lam
        results.append(met)
        print(f"test_acc={met['acc_test']:.4f}, rank={met['eff_rank']}, "
              f"rmse={met['rmse']:.5f}")

    # pick best
    best_i = np.argmax([r['acc_test'] for r in results])
    best_lam = lambdas[best_i]
    print(f"\n  => best lambda = {best_lam} "
          f"(test acc = {results[best_i]['acc_test']:.4f})")
    return results, best_lam


def noise_experiment(m, n, rank, obs_prob, lam, noise_levels,
                     eta=4.0, max_iter=300):
    """See how accuracy changes with different noise levels."""
    print("\n" + "=" * 60)
    print("Noise robustness")
    print("=" * 60)

    results = []
    for np_ in noise_levels:
        print(f"  noise={np_:.2f} ... ", end="", flush=True)
        d = generate_data(m, n, rank, obs_prob, noise_prob=np_, seed=42)
        X_est, _ = fista(d['Y'], d['R_train'], lam,
                         eta=eta, max_iter=max_iter, verbose=False)
        met = evaluate(X_est, d)
        met['noise_prob'] = np_
        results.append(met)
        print(f"test={met['acc_test']:.4f}, all={met['acc_all']:.4f}, "
              f"rank={met['eff_rank']}")
    return results


def obsrate_experiment(m, n, rank, lam, obs_probs,
                       eta=4.0, max_iter=300):
    """See how accuracy varies with observation probability."""
    print("\n" + "=" * 60)
    print("Observation rate analysis")
    print("=" * 60)

    results = []
    for op in obs_probs:
        print(f"  obs_prob={op:.2f} ... ", end="", flush=True)
        d = generate_data(m, n, rank, op, noise_prob=0.0, seed=42)
        X_est, _ = fista(d['Y'], d['R_train'], lam,
                         eta=eta, max_iter=max_iter, verbose=False)
        met = evaluate(X_est, d)
        met['obs_prob'] = op
        results.append(met)
        print(f"test={met['acc_test']:.4f}, unobs={met['acc_unobs']:.4f}, "
              f"rank={met['eff_rank']}")
    return results


# ---- Plotting functions ----

def plot_convergence(hist, met, lam, save_dir="."):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    iters = hist['iters']

    # surrogate objective
    axes[0].plot(iters, hist['surr_obj'], 'b-', lw=1.5,
                 label='Surrogate (logistic + nuclear)')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Objective Value')
    axes[0].set_title('Surrogate Objective Convergence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # L0 objective
    axes[1].plot(iters, hist['l0_obj'], 'r-', lw=1.5,
                 label='Original (L0 + nuclear)')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Objective Value')
    axes[1].set_title('Original L0 Objective Trend')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # relative change (log scale)
    axes[2].semilogy(iters, hist['rel_change'], 'g-', lw=1.5)
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Relative Change (log scale)')
    axes[2].set_title('Convergence (Relative Change in X)')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'FISTA Convergence (lambda = {lam})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig1_convergence.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved fig1_convergence.png")


def plot_svd(hist, data, X_est, save_dir="."):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # compare singular values
    s_est = np.linalg.svd(X_est, compute_uv=False)
    s_true = np.linalg.svd(data['X_star'], compute_uv=False)
    k = min(20, len(s_est))
    axes[0].bar(np.arange(k) - 0.15, s_true[:k], width=0.3, alpha=0.7,
                label='X* (ground truth)', color='steelblue')
    axes[0].bar(np.arange(k) + 0.15, s_est[:k], width=0.3, alpha=0.7,
                label='X_hat (estimated)', color='coral')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Singular Value')
    axes[0].set_title('Singular Values: Ground Truth vs Estimated')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # evolution over iterations
    sv_hist = hist['sv_history']
    n_show = min(8, len(sv_hist[0]))
    sv_mat = np.array([sv[:n_show] for sv in sv_hist])
    for j in range(n_show):
        axes[1].plot(hist['iters'], sv_mat[:, j], lw=1.5, label=f's_{j+1}')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Singular Value')
    axes[1].set_title('Evolution of Top Singular Values')
    axes[1].legend(fontsize=9, ncol=2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig2_singular_values.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved fig2_singular_values.png")


def plot_lambda_results(results, save_dir="."):
    lams = [r['lam'] for r in results]
    test_acc = [r['acc_test'] for r in results]
    train_acc = [r['acc_train'] for r in results]
    ranks = [r['eff_rank'] for r in results]
    rmses = [r['rmse'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].semilogx(lams, test_acc, 'ro-', lw=2, ms=8, label='Held-out (test)')
    axes[0].semilogx(lams, train_acc, 'bs--', lw=2, ms=8, label='Training')
    axes[0].set_xlabel('lambda (log scale)')
    axes[0].set_ylabel('Sign Accuracy')
    axes[0].set_title('Sign Accuracy vs lambda')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(lams, ranks, 'g^-', lw=2, ms=8)
    axes[1].set_xlabel('lambda (log scale)')
    axes[1].set_ylabel('Effective Rank')
    axes[1].set_title('Effective Rank vs lambda')
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogx(lams, rmses, 'md-', lw=2, ms=8)
    axes[2].set_xlabel('lambda (log scale)')
    axes[2].set_ylabel('RMSE')
    axes[2].set_title('RMSE to Ground Truth vs lambda')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Lambda Selection (Grid Search)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig3_lambda_search.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved fig3_lambda_search.png")


def plot_noise_results(results, save_dir="."):
    nps = [r['noise_prob'] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(nps, [r['acc_test'] for r in results], 'ro-', lw=2, ms=8,
                 label='Held-out observed')
    axes[0].plot(nps, [r['acc_all'] for r in results], 'bs--', lw=2, ms=8,
                 label='All entries')
    axes[0].plot(nps, [r['acc_unobs'] for r in results], 'g^:', lw=2, ms=8,
                 label='Unobserved')
    axes[0].set_xlabel('Label Noise Probability')
    axes[0].set_ylabel('Sign Accuracy')
    axes[0].set_title('Accuracy vs Label Noise')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(nps, [r['eff_rank'] for r in results], 'md-', lw=2, ms=8)
    axes[1].set_xlabel('Label Noise Probability')
    axes[1].set_ylabel('Effective Rank')
    axes[1].set_title('Effective Rank vs Label Noise')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Noise Robustness Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig4_noise_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved fig4_noise_analysis.png")


def plot_obsrate_results(results, save_dir="."):
    ops = [r['obs_prob'] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ops, [r['acc_test'] for r in results], 'ro-', lw=2, ms=8,
                 label='Held-out observed')
    axes[0].plot(ops, [r['acc_unobs'] for r in results], 'bs--', lw=2, ms=8,
                 label='Unobserved entries')
    axes[0].set_xlabel('Observation Probability')
    axes[0].set_ylabel('Sign Accuracy')
    axes[0].set_title('Accuracy vs Observation Rate')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ops, [r['eff_rank'] for r in results], 'g^-', lw=2, ms=8)
    axes[1].set_xlabel('Observation Probability')
    axes[1].set_ylabel('Effective Rank')
    axes[1].set_title('Effective Rank vs Observation Rate')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Observation Rate Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig5_obs_rate.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved fig5_obs_rate.png")


def plot_matrices(X_est, data, save_dir="."):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # ground truth
    axes[0].imshow(np.sign(data['X_star']), cmap='RdBu', vmin=-1, vmax=1,
                   aspect='equal')
    axes[0].set_title('sign(X*) Ground Truth')
    axes[0].set_xlabel('Column'); axes[0].set_ylabel('Row')

    # observed entries
    obs = data['R'] * data['Y']
    obs[data['R'] == 0] = 0
    axes[1].imshow(obs, cmap='RdBu', vmin=-1, vmax=1, aspect='equal')
    axes[1].set_title('Observed Y (gray=unobserved)')
    axes[1].set_xlabel('Column')

    # estimated
    se = np.sign(X_est)
    se[se == 0] = 1
    axes[2].imshow(se, cmap='RdBu', vmin=-1, vmax=1, aspect='equal')
    axes[2].set_title('sign(X_hat) Estimated')
    axes[2].set_xlabel('Column')

    # errors
    st = np.sign(data['X_star'])
    st[st == 0] = 1
    err = (se != st).astype(float)
    axes[3].imshow(err, cmap='Reds', vmin=0, vmax=1, aspect='equal')
    axes[3].set_title('Sign Errors (red=mismatch)')
    axes[3].set_xlabel('Column')

    fig.suptitle('Matrix Visualization', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig6_matrices.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  saved fig6_matrices.png")


# ---- Main ----

def main():
    print("=" * 60)
    print("  Binary Matrix Completion - Midsem Exam")
    print("  FISTA + Logistic Loss + SVT")
    print("=" * 60)

    # config
    m, n = 100, 100
    rank = 5
    obs_prob = 0.5
    eta = 4.0    # optimal step size: L_f = 0.25, so eta = 1/L = 4.0
    max_iter = 500
    search_iter = 300

    save_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\n  m={m}, n={n}, rank={rank}, obs_prob={obs_prob}, eta={eta}")

    # generate data
    data = generate_data(m, n, rank, obs_prob, noise_prob=0.0, seed=42)
    print(f"  {data['n_obs']} observed entries "
          f"({100*data['n_obs']/(m*n):.1f}%), "
          f"{data['n_train']} train, {data['n_test']} test")

    # Exp 1: find best lambda
    lambdas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    lam_results, best_lam = lambda_search(data, lambdas,
                                          eta=eta, max_iter=search_iter)
    plot_lambda_results(lam_results, save_dir)

    # Exp 2: convergence with best lambda
    X_est, hist, met, runtime = run_convergence(data, best_lam,
                                                eta=eta, max_iter=max_iter)
    plot_convergence(hist, met, best_lam, save_dir)
    plot_svd(hist, data, X_est, save_dir)
    plot_matrices(X_est, data, save_dir)

    # Exp 3: noise robustness
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    noise_res = noise_experiment(m, n, rank, obs_prob, best_lam,
                                 noise_levels, eta=eta, max_iter=search_iter)
    plot_noise_results(noise_res, save_dir)

    # Exp 4: observation rate
    obs_rates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    obs_res = obsrate_experiment(m, n, rank, best_lam,
                                 obs_rates, eta=eta, max_iter=search_iter)
    plot_obsrate_results(obs_res, save_dir)

    # summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  best lambda:       {best_lam}")
    print(f"  runtime:           {runtime:.2f}s")
    print(f"  iterations:        {hist['iters'][-1]}")
    print(f"  restarts:          {hist.get('n_restarts', 0)}")
    print(f"  train acc:         {met['acc_train']:.4f}")
    print(f"  test acc:          {met['acc_test']:.4f}")
    print(f"  all entries acc:   {met['acc_all']:.4f}")
    print(f"  unobserved acc:    {met['acc_unobs']:.4f}")
    print(f"  rmse:              {met['rmse']:.4f}")
    print(f"  eff rank:          {met['eff_rank']}")
    print(f"  nuc norm X_hat:    {met['nuc_norm']:.2f}")
    print(f"  true rank:         {rank}")
    nuc_star = nuclear_norm(data['X_star'])
    print(f"  nuc norm X*:       {nuc_star:.2f}")
    print(f"\n  figures saved to {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()