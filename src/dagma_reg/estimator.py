# SPDX-License-Identifier: Apache-2.0
"""Standalone observation-to-graph interface; rows are causes, columns effects."""

from __future__ import annotations

import json
import random
from copy import deepcopy
from importlib.resources import files
from numbers import Integral

import numpy as np
import torch

from . import stein
from .dagma_model import DagmaMLP, DagmaNonlinear
from .glasso_cv import glasso_existence
from .penalty import build_lambda


def load_defaults():
    """Return an independent copy of the package's default configuration."""
    return json.loads(files("dagma_reg").joinpath("default.json").read_text())


def is_acyclic(adjacency):
    """Check a directed adjacency using topological elimination."""
    a = np.asarray(adjacency, dtype=bool)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Expected square adjacency")
    degrees = a.sum(axis=0)
    ready = list(np.flatnonzero(degrees == 0))
    visited = 0
    while ready:
        i = ready.pop()
        visited += 1
        for j in np.flatnonzero(a[i]):
            degrees[j] -= 1
            if degrees[j] == 0:
                ready.append(j)
    return visited == len(a)


class DagmaReg:
    """Fit raw observations using the default configuration.

    Parameter dictionaries override the corresponding configuration block. Training
    runs on CPU in float64. ``seed`` seeds Python, NumPy and PyTorch globally;
    ``num_threads`` controls PyTorch's global CPU thread count.
    """

    def __init__(
        self,
        *,
        params=None,
        training=None,
        stein=None,
        glasso=None,
        bootstrap=None,
        threshold=None,
        seed=0,
        num_threads=1,
        trace_every=None,
    ):
        self.config = load_defaults()
        for key, overrides in (
            ("params", params),
            ("training", training),
            ("stein", stein),
            ("glasso", glasso),
            ("bootstrap", bootstrap),
        ):
            if overrides is not None:
                unknown = set(overrides) - self.config[key].keys()
                if unknown:
                    raise ValueError(f"Unknown {key} settings: {sorted(unknown)}")
                self.config[key].update(deepcopy(overrides))
        self.threshold = self.config["threshold"] if threshold is None else float(threshold)
        self.seed = seed
        self.num_threads = num_threads
        self.trace_every = self.config["trace_every"] if trace_every is None else trace_every
        self._validate()

    def _validate(self):
        c, p, t = self.config, self.config["params"], self.config["training"]
        if not np.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("threshold must be finite and nonnegative")
        if (
            not 0 <= p["alpha"] <= 1
            or p["direction"] not in ("binary", "bootstrap")
            or p["gate"] not in ("none", "linear", "floor20percent")
        ):
            raise ValueError("Invalid alpha, direction or gate")
        for k in ("q_G", "q_H"):
            if not np.isfinite(p[k]) or p[k] <= 0:
                raise ValueError(f"{k} must be positive")
        for k in ("lambda_s", "lambda_e", "lambda_d"):
            if not np.isfinite(p[k]) or p[k] < 0:
                raise ValueError(f"{k} must be nonnegative")
        for block, keys in (
            (t, ("hidden_dim", "T", "warm_iter", "max_iter")),
            (c["bootstrap"], ("repetitions", "subsample_size")),
            (c["glasso"], ("cv_folds", "glasso_max_iter")),
        ):
            for k in keys:
                if not isinstance(block[k], Integral) or isinstance(block[k], bool) or block[k] < 1:
                    raise ValueError(f"{k} must be a positive integer")
        if c["glasso"]["cv_folds"] < 2:
            raise ValueError("cv_folds must be at least 2")
        for block, keys in (
            (t, ("lr",)),
            (c["stein"], ("eta_G", "eta_H")),
            (c["glasso"], ("glasso_tol",)),
        ):
            for k in keys:
                if not np.isfinite(block[k]) or block[k] <= 0:
                    raise ValueError(f"{k} must be finite and positive")
        if not np.isfinite(t["lambda2"]) or t["lambda2"] < 0:
            raise ValueError("lambda2 must be nonnegative")
        fractions = np.asarray(c["glasso"]["alpha_scale_fractions"], dtype=float)
        if (
            fractions.ndim != 1
            or not len(fractions)
            or not np.isfinite(fractions).all()
            or np.any(fractions <= 0)
            or np.any(np.diff(fractions) <= 0)
        ):
            raise ValueError("alpha_scale_fractions must be positive and strictly increasing")
        if (
            c["stein"]["order_data"] != "raw"
            or c["glasso"]["representation"] != "correlation"
            or c["bootstrap"]["replace"]
        ):
            raise ValueError(
                "Only raw SCORE, correlation GLasso and sampling without replacement are supported"
            )
        for name, value, minimum in (
            ("num_threads", self.num_threads, 1),
            ("trace_every", self.trace_every, 0),
            ("seed", self.seed, 0),
            ("cv_seed", c["glasso"]["cv_seed"], 0),
            ("bootstrap.seed", c["bootstrap"]["seed"], 0),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.seed >= 2**32 or c["glasso"]["cv_seed"] >= 2**32:
            raise ValueError("seed and cv_seed must be less than 2**32")

    def compute_priors(self, X):
        """Compute label-free priors on raw data, skipping unused subsamples."""
        self._validate()
        X = self._data(X)
        c = self.config
        if c["params"]["direction"] == "bootstrap" and c["bootstrap"]["subsample_size"] > len(X):
            raise ValueError("subsample_size exceeds n; explicitly reduce bootstrap.subsample_size")
        g = glasso_existence(X, c["glasso"])
        eta = {k: c["stein"][k] for k in ("eta_G", "eta_H")}
        order, variance = stein.score_order(X, **eta)
        M = stein.hessian_mean_abs(X, **eta)
        score, rank = stein.existence_from_hessian(M)
        out = {
            **g,
            "order": np.asarray(order),
            "first_diag_variance": variance,
            "binary": stein.order_direction(order),
            "M_raw": M,
            "hessian_score": score,
            "r_H": rank,
        }
        if c["params"]["direction"] == "bootstrap":
            b = c["bootstrap"]
            out["bootstrap_orders"], out["bootstrap"] = stein.bootstrap_orders(
                X,
                **eta,
                repetitions=b["repetitions"],
                subsample_size=b["subsample_size"],
                seed=b["seed"],
            )
        return out

    def _data(self, X):
        if np.iscomplexobj(X):
            raise ValueError("X must contain real-valued observations")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < max(3, self.config["glasso"]["cv_folds"]) or X.shape[1] < 2:
            raise ValueError(
                "X must have shape (n_samples, n_features), n >= max(3, cv_folds), d >= 2"
            )
        if not np.isfinite(X).all():
            raise ValueError("X must be finite with no constant columns")
        with np.errstate(over="ignore", invalid="ignore"):
            scale = X.std(axis=0)
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError(
                "X must have finite, nonzero column standard deviations (no constant columns)"
            )
        folds = self.config["glasso"]["cv_folds"]
        if len(X) - (len(X) + folds - 1) // folds < 2:
            raise ValueError(
                "Each GLasso cross-validation training fold needs at least two samples"
            )
        return X

    def fit(self, X):
        """Fit raw observations and return self; no true graph is accepted."""
        self._validate()
        X = self._data(X)
        torch.set_num_threads(self.num_threads)
        priors = self.compute_priors(X)
        penalty = build_lambda(priors, self.config["params"])
        mean = X.mean(axis=0, keepdims=True)
        scale = X.std(axis=0, keepdims=True, ddof=0)
        Z = ((X - mean) / scale).astype(float)
        random.seed(int(self.seed))
        np.random.seed(int(self.seed))
        torch.manual_seed(int(self.seed))
        train = dict(self.config["training"])
        hidden = train.pop("hidden_dim")
        optimizer = DagmaNonlinear(DagmaMLP([X.shape[1], hidden, 1], bias=True))
        weights = optimizer.fit(Z, penalty, **train, trace_every=self.trace_every)
        if not np.isfinite(weights).all():
            raise FloatingPointError("Optimization returned non-finite weights")
        adjacency = (np.abs(weights) > self.threshold).astype(int)
        np.fill_diagonal(adjacency, 0)
        acyclic = is_acyclic(adjacency)
        last_attempts = {step["outer_step"]: step for step in optimizer.history}
        diagnostics = {
            "steps": deepcopy(optimizer.history),
            "trace": deepcopy(optimizer.trace),
            "retries_h_negative": optimizer.retries,
            "is_acyclic": acyclic,
            "outer_steps_planned": train["T"],
            "all_outer_steps_succeeded": all(
                last_attempts[i]["success"] for i in range(train["T"])
            ),
        }
        self.priors_ = priors
        self.lambda_ = penalty
        self.mean_ = mean
        self.scale_ = scale
        self.optimizer_ = optimizer
        self.weights_ = weights
        self.adjacency_ = adjacency
        self.is_acyclic_ = acyclic
        self.diagnostics_ = diagnostics
        return self

    def get_adjacency(self, threshold=None):
        """Return abs(W) > threshold with no self edges; cycles are retained."""
        if not hasattr(self, "weights_"):
            raise RuntimeError("Call fit before get_adjacency")
        threshold = self.threshold if threshold is None else float(threshold)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative")
        a = (np.abs(self.weights_) > threshold).astype(int)
        np.fill_diagonal(a, 0)
        return a
