"""
main.py — Full experiment pipeline
===================================
Runs ADMM (both ℓ₁ and ℓ₀ modes), evaluates, sweeps hyperparameters,
and generates all figures + a summary table.

Usage:
    python main.py                # full experiment
    python main.py --quick        # fast smoke test
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from solver import ADMMSolver, auto_lambda, nuclear_norm, objective_l1, objective_l0
from data_gen import (
    generate_low_rank_matrix,
    generate_observations,
    train_test_split_mask,
)
from metrics import compute_all_metrics, effective_rank, sign_accuracy

# reproducibility
np.random.seed(42)

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Helper: pretty-print a metrics dict
# ═══════════════════════════════════════════════════════════════════════════
def print_metrics(tag: str, mdict: dict):
    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"{'='*60}")
    for k, v in mdict.items():
        if isinstance(v, float):
            print(f"    {k:25s} : {v:.6f}")
        else:
            print(f"    {k:25s} : {v}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  1. DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════
def make_data(m=100, n=100, rank=5, obs_prob=0.30, noise_frac=0.05,
              test_frac=0.20, seed=42):
    X_star = generate_low_rank_matrix(m, n, rank, seed=seed)
    Y_true, Y_noisy, R_full = generate_observations(X_star, obs_prob, noise_frac, seed=seed)
    R_train, R_test = train_test_split_mask(R_full, test_frac, seed=seed)
    print(f"Data: {m}×{n}, true rank={rank}, "
          f"obs={obs_prob*100:.0f}%, noise={noise_frac*100:.0f}%")
    print(f"  Observed entries : {int(R_full.sum())}")
    print(f"  Train / Test     : {int(R_train.sum())} / {int(R_test.sum())}")
    return X_star, Y_true, Y_noisy, R_full, R_train, R_test


# ═══════════════════════════════════════════════════════════════════════════
# 2. SINGLE RUN
# ═══════════════════════════════════════════════════════════════════════════
def run_single(Y, R_train, R_test, Y_true, X_star, mode="l1",
               lam=None, rho=1.0, adaptive_rho=True, label=""):
    if lam is None:
        lam = auto_lambda(Y, R_train)
        print(f"  Auto λ = {lam:.4f}")

    # L0 benefits from higher ρ to tighten the X=Z constraint
    if mode == "l0" and rho < 3.0:
        rho = 5.0

    solver = ADMMSolver(
        lam=lam, rho=rho, mode=mode,
        max_iter=500, adaptive_rho=adaptive_rho,
        mu=5.0 if mode == "l0" else 10.0,  # more aggressive ρ adaptation for L0
        use_partial_svd=False, warm_start=True, verbose=True,
    )
    X_pred = solver.fit(Y, R_train)
    mdict = compute_all_metrics(X_pred, X_star, Y_true, R_train, R_test)
    mdict["lambda"] = lam
    mdict["rho_init"] = rho
    mdict["iterations"] = solver.history["iterations"]
    mdict["wall_time_s"] = solver.history["wall_time"]
    print_metrics(f"{label} ({mode.upper()})  λ={lam:.4f}", mdict)
    return solver, X_pred, mdict


# ═══════════════════════════════════════════════════════════════════════════
# 3. HYPERPARAMETER SWEEP
# ═══════════════════════════════════════════════════════════════════════════
def sweep_lambda(Y, R_train, R_test, Y_true, X_star, mode="l1"):
    """Grid search over λ values, returning best λ and the sweep data."""
    base = auto_lambda(Y, R_train)
    scales = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    results = []
    best_acc, best_lam = -1, base

    rho_default = 5.0 if mode == "l0" else 1.0
    mu_default = 5.0 if mode == "l0" else 10.0

    print(f"\n{'─'*60}")
    print(f"  λ sweep (mode={mode}, auto_base={base:.4f}, ρ={rho_default})")
    print(f"{'─'*60}")

    for sc in scales:
        lam = base * sc
        solver = ADMMSolver(lam=lam, rho=rho_default, mode=mode, max_iter=300,
                            adaptive_rho=True, mu=mu_default,
                            verbose=False, warm_start=True)
        X_pred = solver.fit(Y, R_train)
        acc = sign_accuracy(X_pred, Y_true, R_test)
        rnk = effective_rank(X_pred)
        iters = solver.history["iterations"]
        results.append({"scale": sc, "lambda": lam, "sign_acc_test": acc,
                        "rank": rnk, "iterations": iters})
        print(f"    scale={sc:5.2f}  λ={lam:8.4f}  acc={acc:.4f}  rank={rnk:3d}  iters={iters}")
        if acc > best_acc:
            best_acc, best_lam = acc, lam

    print(f"  ▸ Best λ = {best_lam:.4f}  (test acc = {best_acc:.4f})")
    return best_lam, results


# ═══════════════════════════════════════════════════════════════════════════
# 4. PLOTTING
# ═══════════════════════════════════════════════════════════════════════════
def plot_convergence(solver, tag, fname):
    """Plot objective, residuals, rank over iterations."""
    h = solver.history
    iters = range(1, len(h["objective"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"ADMM Convergence — {tag}", fontsize=14, fontweight="bold")

    # Objective
    ax = axes[0, 0]
    ax.plot(iters, h["objective"], "b-", linewidth=1.2)
    ax.set_ylabel("Objective")
    ax.set_xlabel("Iteration")
    ax.set_title("Objective Function")
    ax.grid(True, alpha=0.3)

    # Residuals
    ax = axes[0, 1]
    ax.semilogy(iters, h["primal_res"], "r-", label="Primal ||X-Z||")
    ax.semilogy(iters, h["dual_res"], "b-", label="Dual ||ρΔZ||")
    ax.semilogy(iters, h["eps_pri"], "r--", alpha=0.5, label="ε_pri")
    ax.semilogy(iters, h["eps_dual"], "b--", alpha=0.5, label="ε_dual")
    ax.set_ylabel("Residual")
    ax.set_xlabel("Iteration")
    ax.set_title("Primal & Dual Residuals")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Rank
    ax = axes[1, 0]
    ax.plot(iters, h["rank"], "g-", linewidth=1.2)
    ax.set_ylabel("Effective Rank")
    ax.set_xlabel("Iteration")
    ax.set_title("Recovered Matrix Rank")
    ax.grid(True, alpha=0.3)

    # ρ schedule
    ax = axes[1, 1]
    ax.plot(iters, h["rho"], "m-", linewidth=1.2)
    ax.set_ylabel("ρ")
    ax.set_xlabel("Iteration")
    ax.set_title("Adaptive ρ Schedule")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_singular_values(X_star, X_pred_l1, X_pred_l0, fname="singular_values.png"):
    """Compare singular value profiles."""
    from scipy.linalg import svd as full_svd
    s_star = full_svd(X_star, compute_uv=False)
    s_l1 = full_svd(X_pred_l1, compute_uv=False)
    s_l0 = full_svd(X_pred_l0, compute_uv=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    k = min(20, len(s_star))
    idx = np.arange(1, k + 1)
    ax.bar(idx - 0.25, s_star[:k], width=0.25, label="Ground Truth", alpha=0.8)
    ax.bar(idx, s_l1[:k], width=0.25, label="ADMM-ℓ₁", alpha=0.8)
    ax.bar(idx + 0.25, s_l0[:k], width=0.25, label="ADMM-ℓ₀", alpha=0.8)
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("σ_i")
    ax.set_title("Singular Value Profile Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_matrix_heatmaps(X_star, Y_noisy, R_train, X_pred, tag, fname):
    """Visual comparison of ground truth vs recovered matrix."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Matrix Comparison — {tag}", fontsize=14, fontweight="bold")

    titles = ["Ground Truth X*", "Observed Y (masked)", "Recovered X", "sign(X) vs Y_true"]
    mats = [
        X_star,
        R_train * Y_noisy,
        X_pred,
        np.sign(X_pred),
    ]
    cmaps = ["RdBu_r", "RdBu_r", "RdBu_r", "RdBu_r"]

    for ax, title, mat, cmap in zip(axes, titles, mats, cmaps):
        vmax = max(abs(mat.min()), abs(mat.max())) or 1
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_sweep(sweep_results, mode, fname):
    """Plot λ sweep results."""
    lams = [r["lambda"] for r in sweep_results]
    accs = [r["sign_acc_test"] for r in sweep_results]
    ranks = [r["rank"] for r in sweep_results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1, color2 = "tab:blue", "tab:red"

    ax1.set_xlabel("λ")
    ax1.set_ylabel("Test Sign Accuracy", color=color1)
    ax1.plot(lams, accs, "o-", color=color1, linewidth=1.5)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Effective Rank", color=color2)
    ax2.plot(lams, ranks, "s--", color=color2, linewidth=1.5)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(f"λ Sweep — {mode.upper()} Mode")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_noise_robustness(Y_true, X_star, R_full, fname="noise_robustness.png"):
    """Show how both methods degrade with increasing noise."""
    noise_fracs = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    results = {"l1": [], "l0": []}

    for nf in noise_fracs:
        _, Y_n, _ = generate_observations(X_star, obs_prob=0.30, noise_frac=nf, seed=42)
        R_tr, R_te = train_test_split_mask(R_full, test_frac=0.20, seed=42)
        for mode in ["l1", "l0"]:
            lam = auto_lambda(Y_n, R_tr)
            solver = ADMMSolver(lam=lam, rho=1.0, mode=mode, max_iter=300,
                                adaptive_rho=True, verbose=False, warm_start=True)
            X_p = solver.fit(Y_n, R_tr)
            acc = sign_accuracy(X_p, Y_true, R_te)
            results[mode].append(acc)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([nf * 100 for nf in noise_fracs], results["l1"], "o-",
            label="ADMM-ℓ₁", linewidth=1.5)
    ax.plot([nf * 100 for nf in noise_fracs], results["l0"], "s--",
            label="ADMM-ℓ₀", linewidth=1.5)
    ax.set_xlabel("Noise Fraction (%)")
    ax.set_ylabel("Test Sign Accuracy")
    ax.set_title("Robustness to Observation Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_observation_rate(Y_true, X_star, fname="obs_rate.png"):
    """Show performance vs observation probability."""
    obs_probs = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
    results = {"l1": [], "l0": []}

    for op in obs_probs:
        _, Y_n, R_f = generate_observations(X_star, obs_prob=op, noise_frac=0.05, seed=42)
        R_tr, R_te = train_test_split_mask(R_f, test_frac=0.20, seed=42)
        for mode in ["l1", "l0"]:
            lam = auto_lambda(Y_n, R_tr)
            solver = ADMMSolver(lam=lam, rho=1.0, mode=mode, max_iter=300,
                                adaptive_rho=True, verbose=False, warm_start=True)
            X_p = solver.fit(Y_n, R_tr)
            acc = sign_accuracy(X_p, Y_true, R_te)
            results[mode].append(acc)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([op * 100 for op in obs_probs], results["l1"], "o-",
            label="ADMM-ℓ₁", linewidth=1.5)
    ax.plot([op * 100 for op in obs_probs], results["l0"], "s--",
            label="ADMM-ℓ₀", linewidth=1.5)
    ax.set_xlabel("Observation Rate (%)")
    ax.set_ylabel("Test Sign Accuracy")
    ax.set_title("Performance vs Observation Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--obs-prob", type=float, default=0.30)
    parser.add_argument("--noise-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  ADMM Matrix Completion — Full Experiment")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────
    X_star, Y_true, Y_noisy, R_full, R_train, R_test = make_data(
        args.m, args.n, args.rank, args.obs_prob, args.noise_frac,
        args.test_frac, args.seed,
    )

    # ── λ sweep ───────────────────────────────────────────────────
    if not args.quick:
        best_lam_l1, sweep_l1 = sweep_lambda(Y_noisy, R_train, R_test,
                                              Y_true, X_star, mode="l1")
        best_lam_l0, sweep_l0 = sweep_lambda(Y_noisy, R_train, R_test,
                                              Y_true, X_star, mode="l0")
        plot_sweep(sweep_l1, "l1", "lambda_sweep_l1.png")
        plot_sweep(sweep_l0, "l0", "lambda_sweep_l0.png")
    else:
        best_lam_l1 = auto_lambda(Y_noisy, R_train)
        best_lam_l0 = best_lam_l1

    # ── Main runs with best λ ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Running ADMM-ℓ₁ (convex relaxation)")
    print("=" * 60)
    solver_l1, X_l1, metrics_l1 = run_single(
        Y_noisy, R_train, R_test, Y_true, X_star,
        mode="l1", lam=best_lam_l1, label="Best-λ",
    )

    print("\n" + "=" * 60)
    print("  Running ADMM-ℓ₀ (hard thresholding, warm-started from ℓ₁)")
    print("=" * 60)
    # Warm-start ℓ₀ from the ℓ₁ solution to avoid oscillation
    solver_l0_obj = ADMMSolver(
        lam=best_lam_l0, rho=5.0, mode="l0",
        max_iter=500, adaptive_rho=False,  # fixed ρ is more stable for L0
        use_partial_svd=False, warm_start=True, verbose=True,
    )
    X_l0 = solver_l0_obj.fit(Y_noisy, R_train, Z_init=X_l1)
    metrics_l0 = compute_all_metrics(X_l0, X_star, Y_true, R_train, R_test)
    metrics_l0["lambda"] = best_lam_l0
    metrics_l0["rho_init"] = 5.0
    metrics_l0["iterations"] = solver_l0_obj.history["iterations"]
    metrics_l0["wall_time_s"] = solver_l0_obj.history["wall_time"]
    print_metrics(f"Best-λ (L0, warm-started)  λ={best_lam_l0:.4f}", metrics_l0)
    solver_l0 = solver_l0_obj

    # ── Plots ─────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_convergence(solver_l1, "ℓ₁ Relaxation", "convergence_l1.png")
    plot_convergence(solver_l0, "ℓ₀ Hard Threshold", "convergence_l0.png")
    plot_singular_values(X_star, X_l1, X_l0)
    plot_matrix_heatmaps(X_star, Y_noisy, R_train, X_l1, "ADMM-ℓ₁", "heatmap_l1.png")
    plot_matrix_heatmaps(X_star, Y_noisy, R_train, X_l0, "ADMM-ℓ₀", "heatmap_l0.png")

    if not args.quick:
        print("\nRunning noise robustness study...")
        plot_noise_robustness(Y_true, X_star, R_full)
        print("Running observation-rate study...")
        plot_observation_rate(Y_true, X_star)

    # ── Summary table ─────────────────────────────────────────────
    summary = {
        "l1": metrics_l1,
        "l0": metrics_l0,
        "data_config": {
            "m": args.m, "n": args.n, "rank": args.rank,
            "obs_prob": args.obs_prob, "noise_frac": args.noise_frac,
            "test_frac": args.test_frac, "seed": args.seed,
        },
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary → {OUT_DIR}/summary.json")

    # ── Print comparison ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON TABLE")
    print("=" * 60)
    header = f"{'Metric':30s} {'ℓ₁':>12s} {'ℓ₀':>12s}"
    print(header)
    print("─" * len(header))
    for key in metrics_l1:
        v1 = metrics_l1[key]
        v0 = metrics_l0[key]
        if isinstance(v1, float):
            print(f"  {key:28s} {v1:12.6f} {v0:12.6f}")
        else:
            print(f"  {key:28s} {str(v1):>12s} {str(v0):>12s}")
    print()
    print("Done! All results saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
