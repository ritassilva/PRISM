"""Regression checks for GLasso normalization."""

import numpy as np
import pytest

from prism.glasso_cv import partial_correlation


def test_valid_partial_correlations():
    precision = np.array([[4., -1.], [-1., 9.]])
    np.testing.assert_allclose(partial_correlation(precision), [[0., 1 / 6], [1 / 6, 0.]])


@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf])
def test_invalid_precision_diagonal_raises(value):
    with pytest.raises(ValueError):
        partial_correlation(np.array([[value, 0.], [0., 1.]]))


def test_nonfinite_offdiagonal_raises():
    with pytest.raises(ValueError):
        partial_correlation(np.array([[1., np.nan], [np.nan, 1.]]))


@pytest.mark.parametrize("value", [1e-300, 1e300])
def test_invalid_denominator_raises(value):
    with np.errstate(over="ignore", under="ignore"), pytest.raises(ValueError):
        partial_correlation(np.diag([value, value]))
