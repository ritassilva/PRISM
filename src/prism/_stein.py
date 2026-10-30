# SPDX-License-Identifier: MIT
"""Stein estimators adapted from dodiscover.toporder (PyWhy).

Source commit: 6e64456198a5b66e515f250a8f4b143ebc31c9a6.
Only SteinMixin and kernel_width are included; annotations and docstrings
were simplified. Numerical operations and their order are preserved.

MIT License

Copyright (c) 2022 PyWhy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import rbf_kernel


def kernel_width(X: NDArray):
    X_diff = np.expand_dims(X, axis=1) - X
    D = np.linalg.norm(X_diff, axis=2).flatten()
    D_nonzeros = D[D > 0]
    s = np.median(D_nonzeros) if np.any(D_nonzeros) else 1
    return s


class SteinMixin:
    def hessian(self, X: NDArray, eta_G: float, eta_H: float) -> NDArray:
        _, d = X.shape
        s = kernel_width(X)
        K = self._evaluate_kernel(X, s=s)
        nablaK = self._evaluate_nablaK(K, X, s)
        G = self.score(X, eta_G, K, nablaK)
        H = np.stack([self._hessian_col(X, G, col, eta_H, K, s) for col in range(d)], axis=1)
        return H

    def score(
        self,
        X: NDArray,
        eta_G: float,
        K: Optional[NDArray] = None,
        nablaK: Optional[NDArray] = None,
    ) -> NDArray:
        n, _ = X.shape
        if K is None:
            s = kernel_width(X)
            K = self._evaluate_kernel(X, s)
            nablaK = self._evaluate_nablaK(K, X, s)
        G = np.matmul(np.linalg.inv(K + eta_G * np.eye(n)), nablaK)
        return G

    def _hessian_col(
        self, X: NDArray, G: NDArray, c: int, eta: float, K: NDArray, s: float
    ) -> NDArray:
        X_diff = self._X_diff(X)
        n, _, _ = X_diff.shape
        Gv = np.einsum("i,ij->ij", G[:, c], G)
        nabla2vK = np.einsum("ik,ikj,ik->ij", X_diff[:, :, c], X_diff, K) / s**4
        nabla2vK[:, c] -= np.einsum("ik->i", K) / s**2
        H_col = -Gv + np.matmul(np.linalg.inv(K + eta * np.eye(n)), nabla2vK)
        return H_col

    def hessian_diagonal(self, X: NDArray, eta_G: float, eta_H: float) -> NDArray:
        n, _ = X.shape
        s = kernel_width(X)
        K = self._evaluate_kernel(X, s)
        nablaK = self._evaluate_nablaK(K, X, s)
        G = self.score(X, eta_G, K, nablaK)
        X_diff = self._X_diff(X)
        nabla2K = np.einsum("kij,ik->kj", -1 / s**2 + X_diff**2 / s**4, K)
        return -(G**2) + np.matmul(np.linalg.inv(K + eta_H * np.eye(n)), nabla2K)

    def _evaluate_kernel(self, X: NDArray, s: float) -> NDArray:
        K = rbf_kernel(X, gamma=1 / (2 * s**2)) / s
        return K

    def _evaluate_nablaK(self, K: NDArray, X: NDArray, s: float):
        nablaK = -np.einsum("kij,ik->kj", self._X_diff(X), K) / s**2
        return nablaK

    def _X_diff(self, X: NDArray):
        return np.expand_dims(X, axis=1) - X
