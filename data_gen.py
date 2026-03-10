"""
Synthetic data generation for the matrix completion / RPCA experiment.
"""

import numpy as np
from typing import Tuple, Optional


def generate_low_rank_matrix(
    m: int = 100,
    n: int = 100,
    rank: int = 5,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Create a ground-truth low-rank matrix X* = U V^T.

    Parameters
    ----------
    m, n : matrix dimensions
    rank  : true rank
    seed  : random seed for reproducibility

    Returns
    -------
    X_star : (m, n) low-rank matrix
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((m, rank))
    V = rng.standard_normal((n, rank))
    X_star = U @ V.T
    return X_star


def generate_observations(
    X_star: np.ndarray,
    obs_prob: float = 0.30,
    noise_frac: float = 0.05,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From a ground-truth matrix, create binary ratings with noise.

    Parameters
    ----------
    X_star    : ground-truth low-rank matrix
    obs_prob  : probability an entry is observed
    noise_frac: fraction of *observed* entries whose sign is flipped

    Returns
    -------
    Y_true  : (m, n) matrix of ±1 ratings (sign of X_star, before noise)
    Y_noisy : (m, n) matrix of ±1 ratings (with noise on observed entries)
    R       : (m, n) binary observation mask
    """
    rng = np.random.default_rng(seed)
    m, n = X_star.shape

    # True binary ratings
    Y_true = np.sign(X_star)
    Y_true[Y_true == 0] = 1  # break ties

    # Observation mask
    R = (rng.random((m, n)) < obs_prob).astype(float)

    # Introduce noise: flip a fraction of observed entries
    Y_noisy = Y_true.copy()
    observed_indices = np.argwhere(R == 1)
    n_observed = len(observed_indices)
    n_flip = int(noise_frac * n_observed)
    flip_idx = rng.choice(n_observed, size=n_flip, replace=False)
    for idx in flip_idx:
        i, j = observed_indices[idx]
        Y_noisy[i, j] *= -1

    return Y_true, Y_noisy, R


def train_test_split_mask(
    R: np.ndarray,
    test_frac: float = 0.20,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split observed entries into train and test masks.

    Parameters
    ----------
    R         : full observation mask
    test_frac : fraction of observed entries held out for testing

    Returns
    -------
    R_train : training mask
    R_test  : testing mask (disjoint from R_train, subset of R)
    """
    rng = np.random.default_rng(seed)
    observed = np.argwhere(R == 1)
    n_obs = len(observed)
    n_test = int(test_frac * n_obs)

    test_choice = rng.choice(n_obs, size=n_test, replace=False)
    test_set = set(map(tuple, observed[test_choice]))

    R_train = R.copy()
    R_test = np.zeros_like(R)
    for i, j in test_set:
        R_train[i, j] = 0.0
        R_test[i, j] = 1.0

    return R_train, R_test
