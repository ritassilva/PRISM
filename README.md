# DAGMA-Reg

Nonlinear causal discovery with adaptive regularization based on Graphical
Lasso, Stein Hessian estimates and SCORE ordering.

DAGMA-Reg learns a directed graph from a matrix of observations. The package includes a default
configuration; all model hyperparameters can be overridden.

## Installation

Requires Python 3.10 or later.

```bash
git clone https://github.com/ritassilva/DAGMA-Reg.git
cd DAGMA-Reg
python -m pip install .
```

Dependencies: NumPy, PyTorch and scikit-learn. The Stein estimator is included
in the package; no Git installation or dodiscover dependency is required.
Installation from this repository is currently supported; the package has not
yet been published on PyPI.

## Quick start

```python
import numpy as np
from dagma_reg import DagmaReg

rng = np.random.default_rng(7)
X = rng.normal(size=(100, 3))
X[:, 1] += np.sin(X[:, 0])
X[:, 2] += 0.6 * X[:, 1]

model = DagmaReg(seed=0).fit(X)

W = model.weights_       # Continuous connection magnitudes
A = model.adjacency_     # Binary adjacency at the default threshold of 0.1
print(model.is_acyclic_)
```

`X` must contain raw observations, with samples in rows and variables in columns.
Values must be finite and columns must have nonzero variance. At least two
variables and `max(3, cv_folds)` samples are required.
Each cross-validation training fold must contain at least two samples and have
nonzero variance in every column.

In output matrices, `A[i, j]` represents the edge **i → j**. Weights are magnitudes
of MLP connections, not signed causal coefficients. Binary adjacency has no
self edges.

## Configuration

```python
model = DagmaReg(
    params={"lambda_d": 1.0, "gate": "linear"},
    training={"hidden_dim": 16, "max_iter": 2000},
    glasso={"cv_folds": 3},
    threshold=0.2,
    seed=42,
)
model.fit(X)
```

Overrides are merged with the defaults. Inspect the full configuration with
`from dagma_reg import load_defaults; config = load_defaults()`.
Configuration groups are `params` (regularization), `training` (MLP and optimizer),
`stein`, `glasso`, and `bootstrap`. `threshold` defaults to 0.1, `seed` to 0,
`num_threads` to 1, and `trace_every` to 250; set `trace_every=0` to disable
iteration traces. Seeds must be nonnegative integers; `seed` and `glasso.cv_seed`
must be smaller than 2**32. Subsampling has its own seed in `bootstrap`.

| Parameter | Default |
|---|---:|
| `alpha` | 0.501815289490836 |
| `q_G` | 0.7138502771541309 |
| `q_H` | 0.42077299145379465 |
| `lambda_s` | 0.0002756374551908298 |
| `lambda_e` | 0.25319395844003595 |
| `lambda_d` | 2.3175277085961845 |
| `direction` / `gate` | `binary` / `none` |
| `threshold` | 0.1 |
| `hidden_dim` / `T` | 8 / 6 |
| `lr` | 0.0010429441606122837 |
| `lambda2` | 0.0005442554166274262 |
| `warm_iter` / `max_iter` | 3125 / 5000 |

The per-edge regularization is:

```text
e = alpha * r_G**q_G + (1 - alpha) * r_H**q_H
Lambda = lambda_s + lambda_e * (1 - e)
         + lambda_d * gate(e) * (1 - direction)
```

`Lambda` has a zero diagonal. Supported gates are `none` (1), `linear` (e),
and `floor20percent` (0.2 + 0.8e).

The default binary direction uses a single SCORE order. To use subsampling:

```python
model = DagmaReg(
    params={"direction": "bootstrap"},
    bootstrap={"subsample_size": 100, "repetitions": 25, "seed": 7},
)
```

Subsampling is without replacement. Its default sample size is 1000; if that
exceeds the number of observations, explicitly choose a smaller value.
Subsampling is skipped entirely when using binary direction.

## Results

| Attribute or method | Result |
|---|---|
| `weights_` | Continuous MLP connection magnitudes |
| `adjacency_` | Binary graph, using `abs(W) > threshold` |
| `priors_` | Existence ranks, SCORE order and direction signals |
| `lambda_` | Per-edge regularization matrix |
| `diagnostics_` | Optimization history, trace, retries and success indicators |
| `is_acyclic_` | Whether `adjacency_` is acyclic |
| `get_adjacency(threshold=0.2)` | Graph at another threshold, without refitting |
| `compute_priors(X)` | Priors computed independently of model fitting |

The default threshold does not guarantee acyclicity on new observations. Cycles are reported and are not automatically removed.
Completing the iteration budget does not imply convergence; inspect
`diagnostics_` when assessing a fit.

## Data processing and reproducibility

GLasso standardizes its own training folds. SCORE and the Stein Hessian use raw
observations. DAGMA uses column-standardized data with `ddof=0`; fitted means
and scales are available as `mean_` and `scale_`. No true graph is required.

Fitting uses CPU and float64. `seed` sets Python, NumPy and PyTorch global seeds;
`num_threads=1` controls PyTorch's global CPU thread count. The underlying model
also sets PyTorch's global default dtype. Use separate processes for isolated
concurrent fits. Results may differ across dependency versions and platforms.
Stein estimation uses n × n kernel matrices and can be expensive on large samples.
Local release checks used Python 3.11 on macOS with NumPy 1.26.4 and 2.4.6;
other Python versions and operating systems have not yet been tested.

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check src tests examples
python examples/causal_discovery.py
python -m build
python -m twine check dist/*
```

The [example](https://github.com/ritassilva/DAGMA-Reg/blob/main/examples/causal_discovery.py) fits synthetic observations using
the default configuration and prints the estimated graph.

## References and license

DAGMA-Reg is an independent project based on the DAGMA implementation by
Bello, Aragam and Ravikumar (NeurIPS 2022) and Stein estimation code from
dodiscover/PyWhy. SCORE ordering follows Rolland et al. (ICML 2022).
These acknowledgments identify the scientific and software foundations;
they do not imply endorsement by the original authors or projects.

DAGMA-Reg's original code and its DAGMA-derived implementation are licensed
under Apache 2.0. The bundled Stein estimator retains the MIT license and
copyright notice from PyWhy in `_stein.py`. The distribution therefore declares
`Apache-2.0 AND MIT`; this describes its components and does not require choosing
between the two licenses. External dependencies retain their own licenses.
See [LICENSE](https://github.com/ritassilva/DAGMA-Reg/blob/main/LICENSE) and
[NOTICE](https://github.com/ritassilva/DAGMA-Reg/blob/main/NOTICE) for details.
