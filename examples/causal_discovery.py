# SPDX-License-Identifier: Apache-2.0
"""Run with: python examples/causal_discovery.py"""

import numpy as np

from dagma_reg import DagmaReg

# Synthetic observations with the generating graph x1 -> x2 -> x3.
# Replace X with your own raw data: rows are samples, columns are variables.
rng = np.random.default_rng(7)
x1 = rng.normal(size=200)
x2 = 1.5 * np.sin(x1) + rng.normal(scale=0.5, size=200)
x3 = 0.8 * x2 + rng.normal(scale=0.5, size=200)
X = np.column_stack([x1, x2, x3])

# Use the default hyperparameters. Optional overrides:
# DagmaReg(seed=0, params={"lambda_d": 1.0}, threshold=0.2)
model = DagmaReg(seed=0).fit(X)

# Rows are causes, columns are effects. Weights are unsigned MLP magnitudes.
np.set_printoptions(precision=3, suppress=True)
print("Connection magnitudes:\n", model.weights_)
print("Adjacency (x1, x2, x3):\n", model.adjacency_)
print("Acyclic:", model.is_acyclic_)

# Additional outputs: model.priors_, model.lambda_, model.diagnostics_
# Another threshold without refitting: model.get_adjacency(threshold=0.2)
