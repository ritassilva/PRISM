# SPDX-License-Identifier: Apache-2.0
"""Regression against dodiscover commit 6e64456198a5b66e515f250a8f4b143ebc31c9a6.

Expected outputs were computed with the upstream estimator, independently of
the bundled implementation. No dodiscover installation is needed to run tests.
"""

import numpy as np

from prism._stein import SteinMixin, kernel_width


def test_upstream_stein_regression():
    x = np.array([[0.0, 1.0], [1.0, -0.5], [2.0, 0.7], [-1.0, 2.0], [0.5, -1.0], [-0.7, 0.3]])
    estimator = SteinMixin()
    np.testing.assert_allclose(
        estimator.score(x, eta_G=0.001),
        np.array(
            [
                [-1.393652054028301, -2.170494548413515],
                [-1.3915415145109389, 0.6677470847211083],
                [-0.6921737332746224, -0.5552729835227732],
                [0.45854967784801737, -0.5530276956281691],
                [0.6835023529324118, 1.0589854762181505],
                [2.4474395041013146, 1.5114459196312822],
            ]
        ),
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        estimator.hessian(x, eta_G=0.001, eta_H=0.001),
        np.array(
            [
                [
                    [-4.33576791187471, -2.8990074567236506],
                    [-2.8990074567236506, -4.341948179342293],
                ],
                [[-2.4983167306996, -1.572236135260705], [-1.572236135260705, -3.8581123538541022]],
                [
                    [0.7157202879980649, 0.6671174515390247],
                    [0.6671174515390247, 0.1210173971393877],
                ],
                [
                    [0.0609999915137209, -0.3125395207486847],
                    [-0.3125395207486847, 0.15087579298617992],
                ],
                [
                    [-1.0866677722856402, 0.8901739564040059],
                    [0.8901739564040059, 2.021219057026853],
                ],
                [
                    [-4.475247369500602, -3.4318905212470625],
                    [-3.4318905212470625, -3.8582081919420013],
                ],
            ]
        ),
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        estimator.hessian_diagonal(x, eta_G=0.001, eta_H=0.001),
        np.array(
            [
                [-4.335767911874712, -4.341948179342296],
                [-2.4983167306995973, -3.8581123538541036],
                [0.715720287998066, 0.12101739713938875],
                [0.060999991513721125, 0.15087579298618042],
                [-1.0866677722856428, 2.021219057026852],
                [-4.475247369500599, -3.8582081919419995],
            ]
        ),
        rtol=1e-9,
        atol=1e-10,
    )


def test_kernel_width_duplicate_rows():
    assert kernel_width(np.ones((5, 2))) == 1
    assert kernel_width(np.array([[0.0], [0.0], [2.0]])) == 2
