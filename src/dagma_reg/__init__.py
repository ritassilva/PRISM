# SPDX-License-Identifier: Apache-2.0
"""DAGMA-Reg: adaptive regularization for nonlinear causal discovery."""

from .estimator import DagmaReg, is_acyclic, load_defaults
from .penalty import build_lambda

__version__ = "0.1.0"
__all__ = ["DagmaReg", "build_lambda", "is_acyclic", "load_defaults"]
