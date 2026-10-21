# SPDX-License-Identifier: Apache-2.0
"""Stein-Hessian existence priors and SCORE ordering.

SCORE iteratively removes the node with the smallest variance of the Hessian
diagonal, re-estimating on the remaining variables. The returned order runs
from roots to leaves. Existence scores use mean absolute Hessian entries,
symmetrized and normalized by the Hessian diagonal, then ranked over pairs.
The bundled Stein estimator derives from dodiscover; see _stein.py for its
MIT license and NOTICE for scientific references.
"""

from __future__ import annotations

import numpy as np

from ._stein import SteinMixin


def _estimator(eta_G: float, eta_H: float):
    return SteinMixin()


# --------------------------------------------------------------------------- #
# Order                                                                        #
# --------------------------------------------------------------------------- #
def score_order(X: np.ndarray, eta_G: float, eta_H: float) -> tuple[list[int], np.ndarray]:
    """SCORE leaf peeling with the Hessian diagonal re-estimated at every step.

    Returns (order from root to leaf, variance of the diagonal of every node in
    the first, all-node estimate).
    """
    estimator = _estimator(eta_G, eta_H)
    n, d = X.shape
    remaining = list(range(d))
    sink_first: list[int] = []
    first_variance = None
    while len(remaining) > 1:
        diag = estimator.hessian_diagonal(X[:, remaining], eta_G, eta_H)  # [n, k]
        variance = diag.var(axis=0)
        if first_variance is None:
            first_variance = np.empty(d)
            first_variance[remaining] = variance
        leaf = int(np.argmin(variance))
        sink_first.append(remaining.pop(leaf))
    sink_first.append(remaining[0])
    if first_variance is None:
        first_variance = np.zeros(d)
    return sink_first[::-1], first_variance


def order_full_hessian(X: np.ndarray, eta_G: float, eta_H: float) -> list[int]:
    """SCORE ordering using the full Hessian at every step.

    Provides a reference for checking the diagonal-only implementation.
    """
    estimator = _estimator(eta_G, eta_H)
    remaining = list(range(X.shape[1]))
    sink_first: list[int] = []
    while len(remaining) > 1:
        H = estimator.hessian(X[:, remaining], eta_G=eta_G, eta_H=eta_H)  # [n, k, k]
        variance = np.diagonal(H, axis1=1, axis2=2).var(axis=0)
        sink_first.append(remaining.pop(int(np.argmin(variance))))
    sink_first.append(remaining[0])
    return sink_first[::-1]


def order_direction(order) -> np.ndarray:
    """d[i, j] = 1 if i precedes j (root to leaf), else 0; diagonal 0.5.

    Rows are sources, columns targets: d[i, j] = 1 favours i -> j.
    """
    order = np.asarray(order, dtype=int)
    if sorted(order.tolist()) != list(range(len(order))):
        raise ValueError("Invalid order: not a permutation")
    position = np.empty(len(order), dtype=int)
    position[order] = np.arange(len(order))
    d = (position[:, None] < position[None, :]).astype(float)
    np.fill_diagonal(d, 0.5)
    return d


def bootstrap_orders(
    X: np.ndarray, eta_G: float, eta_H: float, repetitions: int, subsample_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Orders on subsamples drawn WITHOUT replacement, and their vote matrix.

    Returns (orders [repetitions, d], d_bootstrap[i, j] = fraction of orders in
    which i precedes j; diagonal 0.5). d_bootstrap + d_bootstrap.T = 1.
    """
    n, d = X.shape
    if subsample_size > n:
        raise ValueError(f"subsample_size {subsample_size} > n {n}")
    rng = np.random.default_rng(seed)
    orders = np.empty((repetitions, d), dtype=int)
    votes = np.zeros((d, d))
    for repetition in range(repetitions):
        rows = rng.choice(n, size=subsample_size, replace=False)
        order, _ = score_order(X[rows], eta_G, eta_H)
        orders[repetition] = order
        votes += order_direction(order)
    return orders, votes / repetitions


# --------------------------------------------------------------------------- #
# Existence score H                                                            #
# --------------------------------------------------------------------------- #
def hessian_mean_abs(X: np.ndarray, eta_G: float, eta_H: float) -> np.ndarray:
    """mean_n |H(x_n)| with the diagonal kept, [d, d]."""
    H = _estimator(eta_G, eta_H).hessian(X, eta_G=eta_G, eta_H=eta_H)  # [n, d, d]
    if not np.isfinite(H).all():
        raise ValueError("Stein Hessian contains non-finite values")
    return np.abs(H).mean(axis=0)


def rank01_symmetric(raw: np.ndarray) -> np.ndarray:
    """Average-tie ranks over unordered pairs, copied symmetrically to [0, 1]."""
    raw = np.asarray(raw, dtype=float)
    d = raw.shape[0]
    upper = np.triu_indices(d, 1)
    values = raw[upper]
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    ranks = ranks / max(len(values) - 1, 1)
    out = np.zeros_like(raw, dtype=float)
    out[upper] = ranks
    out[(upper[1], upper[0])] = ranks
    return out


def existence_from_hessian(M: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Return the diagonal-normalized symmetric score and its ranks in [0, 1]."""
    M = np.abs(np.asarray(M, dtype=float))
    diag = np.diag(M)
    if np.any(diag <= eps):
        raise ValueError("Hessian diagonal is zero: the full mean |H| is required")
    score = 0.5 * (M + M.T) / np.sqrt(np.outer(diag, diag))
    np.fill_diagonal(score, 0.0)
    return score, rank01_symmetric(score)
