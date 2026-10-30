# SPDX-License-Identifier: Apache-2.0
"""Per-edge regularization from existence and direction priors.

    e = alpha * r_G**q_G + (1 - alpha) * r_H**q_H
    Lambda = lambda_s + lambda_e * (1 - e)
             + lambda_d * gate(e) * (1 - direction)

Rows represent sources and columns targets. The diagonal is zero. Direction
values favor edges consistent with the SCORE order. Existence ranks lie in
[0, 1]; gates are none (1), linear (e), or floor20percent (0.2 + 0.8e).
"""

from __future__ import annotations

import numpy as np

GATES = ("none", "linear", "floor20percent")
DIRECTIONS = ("binary", "bootstrap")


def gate(e: np.ndarray, name: str) -> np.ndarray:
    if name == "none":
        return np.ones_like(e)
    if name == "linear":
        return e
    if name == "floor20percent":
        return 0.2 + 0.8 * e
    raise ValueError(f"Unknown gate {name!r}")


def existence(priors: dict, alpha: float, q_G: float, q_H: float) -> np.ndarray:
    e = alpha * priors["r_G"] ** q_G + (1.0 - alpha) * priors["r_H"] ** q_H
    np.fill_diagonal(e, 0.0)
    return e


def build_lambda(priors: dict, p: dict) -> np.ndarray:
    """Lambda for one parameter set p (keys: lambda_s, lambda_e, lambda_d,
    alpha, q_G, q_H, gate, direction)."""
    if p["direction"] not in DIRECTIONS or p["gate"] not in GATES:
        raise ValueError("Unknown direction or gate")
    if not np.isfinite(p["alpha"]) or not 0 <= p["alpha"] <= 1:
        raise ValueError("alpha must be in [0, 1]")
    for key in ("q_G", "q_H"):
        if not np.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("lambda_s", "lambda_e", "lambda_d"):
        if not np.isfinite(p[key]) or p[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    matrices = {}
    for key in ("r_G", "r_H", p["direction"]):
        value = np.asarray(priors[key], dtype=float)
        if value.ndim != 2 or value.shape[0] != value.shape[1] or len(value) < 2:
            raise ValueError(f"{key} must be a square matrix with at least two variables")
        if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError(f"{key} must contain finite values in [0, 1]")
        matrices[key] = value
    if len({value.shape for value in matrices.values()}) != 1:
        raise ValueError("Prior matrices must have the same shape")
    priors = matrices
    e = existence(priors, p["alpha"], p["q_G"], p["q_H"])
    d = priors[p["direction"]]
    Lambda = (
        p["lambda_s"] + p["lambda_e"] * (1.0 - e) + p["lambda_d"] * gate(e, p["gate"]) * (1.0 - d)
    )
    np.fill_diagonal(Lambda, 0.0)
    if not np.isfinite(Lambda).all() or np.any(Lambda < 0):
        raise ValueError("Invalid Lambda: non-finite or negative")
    return Lambda
