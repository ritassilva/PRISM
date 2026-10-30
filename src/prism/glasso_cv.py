# SPDX-License-Identifier: Apache-2.0
"""Graphical Lasso existence priors selected without graph labels.

The regularization grid scales with the largest absolute off-diagonal
covariance of standardized training observations. Cross-validation scores
held-out Gaussian log-likelihood; the one-standard-error rule selects the
largest eligible regularization with a valid full-data refit. Absolute
partial correlations are converted to symmetric average-tie ranks.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
from sklearn.covariance import GraphicalLasso
from sklearn.model_selection import KFold

from .stein import rank01_symmetric


def _standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Cannot standardise a constant or non-finite variable")
    return (train - mean) / scale, (test - mean) / scale


def _offdiag_scale(X: np.ndarray) -> float:
    covariance = np.cov(X, rowvar=False, ddof=0)
    values = np.abs(covariance[~np.eye(covariance.shape[0], dtype=bool)])
    scale = float(values.max(initial=0.0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Non-positive off-diagonal covariance scale")
    return scale


def _fit(X: np.ndarray, alpha: float, cfg: dict) -> GraphicalLasso:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GraphicalLasso(
            alpha=float(alpha), max_iter=int(cfg["glasso_max_iter"]), tol=float(cfg["glasso_tol"])
        ).fit(X)
    if not np.isfinite(model.precision_).all():
        raise FloatingPointError("Non-finite precision matrix")
    return model


def partial_correlation(precision: np.ndarray) -> np.ndarray:
    """Absolute partial correlations; reject invalid normalization references."""
    precision = np.asarray(precision, dtype=float)
    if not np.isfinite(precision).all():
        raise ValueError("GLasso precision matrix contains non-finite values")
    diagonal = np.diag(precision)
    if np.any(diagonal <= 0):
        raise ValueError("GLasso precision diagonal must be strictly positive")
    denominator = np.sqrt(np.outer(diagonal, diagonal))
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0):
        raise ValueError("GLasso partial-correlation denominator must be finite and positive")
    score = np.abs(precision) / denominator
    if not np.isfinite(score).all():
        raise ValueError("GLasso partial correlations contain non-finite values")
    np.fill_diagonal(score, 0.0)
    return score


def _one_se_refittable(means, standard_errors, valid_folds, refittable_indices) -> tuple[int, int]:
    eligible_best = np.flatnonzero(valid_folds == valid_folds.max())
    if eligible_best.size == 0 or valid_folds[eligible_best[0]] == 0:
        raise ValueError("No valid CV score")
    best = int(eligible_best[np.nanargmax(means[eligible_best])])
    eligible = np.flatnonzero(
        (valid_folds == valid_folds.max()) & (means >= means[best] - standard_errors[best])
    )
    provisional = int(eligible[-1])

    full_valid = set(int(i) for i in refittable_indices)
    refittable = np.asarray(
        [i for i in range(len(means)) if valid_folds[i] == valid_folds.max() and i in full_valid],
        dtype=int,
    )
    if refittable.size == 0:
        raise FloatingPointError("No full-data refit is valid for the CV-supported alphas")
    best_refittable = int(refittable[np.nanargmax(means[refittable])])
    cutoff = means[best_refittable] - standard_errors[best_refittable]
    chosen = refittable[means[refittable] >= cutoff]
    if chosen.size == 0:
        raise FloatingPointError("No refittable alpha satisfies the one-standard-error rule")
    return provisional, int(chosen[-1])


def glasso_existence(X_raw: np.ndarray, cfg: dict) -> dict:
    """CV-selected GLasso on the correlation representation of X_raw."""
    fractions = np.asarray(cfg["alpha_scale_fractions"], dtype=float)
    folds = int(cfg["cv_folds"])
    splitter = KFold(n_splits=folds, shuffle=True, random_state=int(cfg["cv_seed"]))
    fold_scores = np.full((folds, len(fractions)), np.nan)
    errors = []
    for fold, (train_index, test_index) in enumerate(splitter.split(X_raw)):
        train, test = _standardise(X_raw[train_index], X_raw[test_index])
        scale = _offdiag_scale(train)
        for k, fraction in enumerate(fractions):
            try:
                fold_scores[fold, k] = float(_fit(train, fraction * scale, cfg).score(test))
            except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
                errors.append(
                    {
                        "stage": "cv",
                        "fold": fold,
                        "fraction": float(fraction),
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

    valid_folds = np.isfinite(fold_scores).sum(axis=0)
    means = np.full(len(fractions), np.nan)
    standard_errors = np.full(len(fractions), np.nan)
    for k in range(len(fractions)):
        values = fold_scores[:, k][np.isfinite(fold_scores[:, k])]
        if values.size:
            means[k] = float(values.mean())
            standard_errors[k] = (
                float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
            )

    full, _ = _standardise(X_raw, X_raw)
    full_alphas = fractions * _offdiag_scale(full)
    models = {}
    densities = np.full(len(fractions), np.nan)
    d = full.shape[1]
    for k, alpha in enumerate(full_alphas):
        try:
            model = _fit(full, float(alpha), cfg)
            models[k] = model
            offdiag = model.precision_.copy()
            np.fill_diagonal(offdiag, 0.0)
            edges = int(np.count_nonzero(np.abs(offdiag) > 1e-10) // 2)
            densities[k] = 2 * edges / (d * (d - 1))
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
            errors.append(
                {
                    "stage": "full",
                    "fraction": float(fractions[k]),
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    if not models:
        raise FloatingPointError("No valid full-data GLasso solution")

    provisional, chosen = _one_se_refittable(
        means, standard_errors, valid_folds, np.asarray(sorted(models), dtype=int)
    )
    score = partial_correlation(models[chosen].precision_)
    return {
        "glasso_partial_correlation": score,
        "r_G": rank01_symmetric(score),
        "glasso_fraction": float(fractions[chosen]),
        "glasso_alpha": float(full_alphas[chosen]),
        "glasso_density": float(densities[chosen]),
        "glasso_refit_adjusted": bool(chosen != provisional),
        "glasso_cv_means": means,
        "glasso_errors_json": json.dumps(errors, sort_keys=True),
    }
