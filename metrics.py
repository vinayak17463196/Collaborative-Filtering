"""
Evaluation metrics for the matrix completion task.
"""

import numpy as np
from scipy.linalg import svd
from typing import Dict


def sign_accuracy(X_pred: np.ndarray, Y_true: np.ndarray, mask: np.ndarray) -> float:
    """
    Fraction of entries in `mask` where sign(X_pred) == Y_true.
    """
    sel = mask == 1
    if sel.sum() == 0:
        return float("nan")
    preds = np.sign(X_pred[sel])
    preds[preds == 0] = 1
    return float(np.mean(preds == Y_true[sel]))


def mse(X_pred: np.ndarray, X_star: np.ndarray, mask: np.ndarray) -> float:
    """Mean Squared Error on masked entries."""
    sel = mask == 1
    if sel.sum() == 0:
        return float("nan")
    return float(np.mean((X_pred[sel] - X_star[sel]) ** 2))


def rmse(X_pred: np.ndarray, X_star: np.ndarray, mask: np.ndarray) -> float:
    """Root Mean Squared Error on masked entries."""
    return float(np.sqrt(mse(X_pred, X_star, mask)))


def relative_error(X_pred: np.ndarray, X_star: np.ndarray) -> float:
    """||X_pred - X_star||_F / ||X_star||_F"""
    denom = np.linalg.norm(X_star, "fro")
    if denom < 1e-12:
        return float("inf")
    return float(np.linalg.norm(X_pred - X_star, "fro") / denom)


def effective_rank(X: np.ndarray, tol: float = 1e-6) -> int:
    """Count singular values above tolerance."""
    s = svd(X, compute_uv=False)
    return int(np.sum(s > tol))


def ndcg_at_k(X_pred: np.ndarray, Y_true: np.ndarray, R_test: np.ndarray,
              k: int = 10) -> float:
    """
    Normalised Discounted Cumulative Gain @ k  (per-user, averaged).

    For each user (row), rank items by predicted score. Compute DCG using
    ground-truth relevance (Y_true == +1 → relevant).
    Only evaluated on users with at least one test entry.
    """
    m, n = X_pred.shape
    ndcg_scores = []

    for i in range(m):
        test_items = np.where(R_test[i] == 1)[0]
        if len(test_items) == 0:
            continue

        # rank all items by predicted score
        ranked = np.argsort(-X_pred[i])[:k]
        # relevance: 1 if Y_true == +1, else 0
        gains = np.array([(1.0 if j in test_items and Y_true[i, j] == 1 else 0.0)
                          for j in ranked])
        discounts = np.log2(np.arange(2, k + 2))
        dcg = np.sum(gains / discounts)

        # ideal: sort test relevant items first
        ideal_gains = sorted(
            [1.0 if Y_true[i, j] == 1 else 0.0 for j in test_items],
            reverse=True,
        )[:k]
        if len(ideal_gains) < k:
            ideal_gains += [0.0] * (k - len(ideal_gains))
        idcg = np.sum(np.array(ideal_gains) / discounts[:len(ideal_gains)])

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def precision_at_k(X_pred: np.ndarray, Y_true: np.ndarray,
                   R_test: np.ndarray, k: int = 10) -> float:
    """
    Precision@k averaged over users.
    """
    m, n = X_pred.shape
    precisions = []
    for i in range(m):
        test_items = set(np.where(R_test[i] == 1)[0])
        if not test_items:
            continue
        ranked = np.argsort(-X_pred[i])[:k]
        hits = sum(1 for j in ranked if j in test_items and Y_true[i, j] == 1)
        precisions.append(hits / k)
    return float(np.mean(precisions)) if precisions else 0.0


def compute_all_metrics(
    X_pred: np.ndarray,
    X_star: np.ndarray,
    Y_true: np.ndarray,
    R_train: np.ndarray,
    R_test: np.ndarray,
) -> Dict[str, float]:
    """Compute a full suite of metrics and return as a dictionary."""
    return {
        "sign_acc_train": sign_accuracy(X_pred, Y_true, R_train),
        "sign_acc_test": sign_accuracy(X_pred, Y_true, R_test),
        "sign_acc_unobs": sign_accuracy(X_pred, Y_true, 1 - R_train - R_test),
        "mse_test": mse(X_pred, X_star, R_test),
        "rmse_test": rmse(X_pred, X_star, R_test),
        "relative_error": relative_error(X_pred, X_star),
        "effective_rank": effective_rank(X_pred),
        "ndcg@10": ndcg_at_k(X_pred, Y_true, R_test, k=10),
        "precision@10": precision_at_k(X_pred, Y_true, R_test, k=10),
    }
