# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from dagma_reg import DagmaReg, build_lambda, is_acyclic, load_defaults
from dagma_reg.stein import order_direction


def observations():
    x = np.random.default_rng(7).normal(size=(35, 3))
    x[:, 1] += 0.8 * x[:, 0]
    return x


def test_defaults_and_isolation():
    c = load_defaults()
    assert c["training"]["warm_iter"] == 3125
    assert c["params"]["lambda_d"] == 2.3175277085961845
    c["params"]["alpha"] = -1
    assert DagmaReg().config["params"]["alpha"] == 0.501815289490836


def test_orientation_penalty_and_cycles():
    direction = order_direction([2, 0, 1])
    assert direction[2, 0] == direction[0, 1] == 1
    priors = {"r_G": np.zeros((3, 3)), "r_H": np.ones((3, 3)), "binary": direction}
    p = load_defaults()["params"]
    lam = build_lambda(priors, p)
    expected = p["lambda_s"] + p["lambda_e"] * p["alpha"] + p["lambda_d"] * (1 - direction)
    np.fill_diagonal(expected, 0)
    np.testing.assert_array_equal(lam, expected)
    assert is_acyclic(direction == 1)
    assert not is_acyclic(np.array([[0, 1], [1, 0]]))


def test_end_to_end_and_repeatability(monkeypatch):
    from dagma_reg import stein

    def unused(*args, **kwargs):
        raise AssertionError("Binary direction must not compute subsamples")

    monkeypatch.setattr(stein, "bootstrap_orders", unused)
    kwargs = dict(training={"T": 2, "warm_iter": 5, "max_iter": 8}, trace_every=2)
    a = DagmaReg(**kwargs).fit(observations())
    b = DagmaReg(**kwargs).fit(observations())
    np.testing.assert_array_equal(a.weights_, b.weights_)
    assert a.weights_.shape == (3, 3)
    assert np.diag(a.lambda_).sum() == np.diag(a.adjacency_).sum() == 0
    assert a.diagnostics_["all_outer_steps_succeeded"]
    assert a.diagnostics_["trace"]
    np.testing.assert_array_equal(a.adjacency_, a.get_adjacency())
    assert a.is_acyclic_ == is_acyclic(a.adjacency_)


def test_subsampling_and_invalid_data():
    model = DagmaReg(
        params={"direction": "bootstrap"}, bootstrap={"subsample_size": 20, "repetitions": 2}
    )
    priors = model.compute_priors(observations())
    assert priors["bootstrap_orders"].shape == (2, 3)
    np.testing.assert_array_equal(priors["bootstrap"] + priors["bootstrap"].T, np.ones((3, 3)))
    with pytest.raises(ValueError, match="subsample_size"):
        DagmaReg(params={"direction": "bootstrap"}).compute_priors(observations())
    with pytest.raises(ValueError, match="constant"):
        DagmaReg().fit(np.ones((10, 3)))
    with pytest.raises(ValueError):
        DagmaReg(training={"T": 0})
    with pytest.raises(ValueError):
        DagmaReg(threshold=-1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": 2**32},
        {"seed": 1.5},
        {"num_threads": 1.5},
        {"trace_every": -1},
        {"trace_every": True},
        {"glasso": {"cv_seed": -1}},
        {"bootstrap": {"seed": -1}},
    ],
)
def test_invalid_integer_settings(kwargs):
    with pytest.raises(ValueError):
        DagmaReg(**kwargs)


def test_invalid_observations():
    with pytest.raises(ValueError, match="real-valued"):
        DagmaReg().fit(observations().astype(complex) + 1j)
    with pytest.raises(ValueError, match="finite"):
        DagmaReg().fit(np.full((10, 3), np.nan))
    with pytest.raises(ValueError, match="two samples"):
        DagmaReg(glasso={"cv_folds": 2}).fit(observations()[:3])


def test_subsample_size_checked_before_priors(monkeypatch):
    import dagma_reg.estimator as estimator

    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid sample size must be caught before computing priors")

    monkeypatch.setattr(estimator, "glasso_existence", unexpected)
    with pytest.raises(ValueError, match="subsample_size"):
        DagmaReg(params={"direction": "bootstrap"}).compute_priors(observations())


def test_failed_refit_keeps_previous_result(monkeypatch):
    import dagma_reg.estimator as estimator

    model = DagmaReg(training={"T": 1, "max_iter": 5}).fit(observations())
    previous = (model.priors_, model.lambda_, model.weights_, model.diagnostics_)

    def fail(*args, **kwargs):
        raise FloatingPointError("Fit failed")

    monkeypatch.setattr(estimator.DagmaNonlinear, "fit", fail)
    with pytest.raises(FloatingPointError):
        model.fit(observations() * 2)
    assert all(
        a is b
        for a, b in zip(
            previous, (model.priors_, model.lambda_, model.weights_, model.diagnostics_)
        )
    )


def test_public_penalty_rejects_broadcasting():
    priors = {"r_G": np.zeros((3, 3)), "r_H": np.zeros((2, 2)), "binary": np.zeros((3, 3))}
    with pytest.raises(ValueError, match="same shape"):
        build_lambda(priors, load_defaults()["params"])
