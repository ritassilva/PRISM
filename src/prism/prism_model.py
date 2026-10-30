# SPDX-License-Identifier: Apache-2.0
"""Nonlinear DAGMA with adaptive per-edge regularization.

Based on DAGMA by Bello, Aragam and Ravikumar (NeurIPS 2022); see NOTICE.
Modified for PRISM: adaptive per-edge group L1 regularization,
optimization history and traces, and removal of experiment-specific code.
Uses a zero-initialized first layer, log-MSE loss, log-determinant acyclicity,
adaptive group L1 penalties, and Adam with betas (0.99, 0.999). The central
path reduces mu at each outer step; negative h triggers a retry with a
halved learning rate. Lambda is active from the first outer step.
"""

from __future__ import annotations

import copy
import math
import typing

import numpy as np
import torch
import torch.nn as nn
from torch import optim


class LocallyConnected(nn.Module):
    """Independent linear transformations for each node."""

    def __init__(
        self, num_linear: int, input_features: int, output_features: int, bias: bool = True
    ):
        super().__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(num_linear, input_features, output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        bound = math.sqrt(1.0 / self.input_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(input.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            out += self.bias
        return out


class PrismMLP(nn.Module):
    """Structural equations for the causal graph, one MLP per node."""

    def __init__(
        self, dims: typing.List[int], bias: bool = True, dtype: torch.dtype = torch.double
    ):
        torch.set_default_dtype(dtype)
        super().__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        self.dims, self.d = dims, dims[0]
        self.I = torch.eye(self.d)
        self.fc1 = nn.Linear(self.d, self.d * dims[1], bias=bias)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        layers = []
        for layer in range(len(dims) - 2):
            layers.append(LocallyConnected(self.d, dims[layer + 1], dims[layer + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = x.view(-1, self.dims[0], self.dims[1])
        for fc in self.fc2:
            x = torch.sigmoid(x)
            x = fc(x)
        return x.squeeze(dim=2)

    def h_func(self, s: float = 1.0) -> torch.Tensor:
        fc1_weight = self.fc1.weight.view(self.d, -1, self.d)
        A = torch.sum(fc1_weight**2, dim=1).t()
        return -torch.slogdet(s * self.I - A)[1] + self.d * np.log(s)

    def fc1_l1_reg_adaptive(self, lambda_matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        fc1_weight = self.fc1.weight.view(self.d, -1, self.d)
        A = torch.sum(fc1_weight**2, dim=1).t()  # [i, j]
        return torch.sum(lambda_matrix.to(self.fc1.weight.device) * torch.sqrt(A + eps))

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:
        fc1_weight = self.fc1.weight.view(self.d, -1, self.d)
        A = torch.sum(fc1_weight**2, dim=1).t()
        return torch.sqrt(A).cpu().numpy()


class PrismNonlinear:
    def __init__(self, model: PrismMLP, dtype: torch.dtype = torch.double):
        self.model = model
        self.dtype = dtype
        self.retries = 0
        # Optimisation diagnostics, one record per outer step. Recording only:
        # nothing here enters the loss, the gradients or the schedule.
        self.history: list[dict] = []
        self.trace: list[dict] = []
        self._step: dict = {}
        self._outer: int = 0
        self._attempt: int = 0

    @staticmethod
    def log_mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n, d = target.shape
        return 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))

    def minimize(
        self,
        max_iter: int,
        lr: float,
        lambda2: float,
        mu: float,
        s: float,
        lambda_matrix: torch.Tensor,
        lr_decay: bool = False,
        tol: float = 1e-6,
        trace_every: int = 0,
    ) -> bool:
        """Minimize the objective with a fixed per-edge penalty matrix.

        ``trace_every`` > 0 records the objective, its parts, h and the norms of
        W every that many iterations, for the convergence plots. Recording only.
        """
        optimizer = optim.Adam(
            self.model.parameters(), lr=lr, betas=(0.99, 0.999), weight_decay=mu * lambda2
        )
        if lr_decay is True:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        obj_prev = 1e16
        self._step = {
            "budget": int(max_iter),
            "iterations": int(max_iter),
            "stop": "budget",
            "lr": lr,
            "lr_decay": bool(lr_decay),
            "mu": mu,
            "s": s,
            "tol": tol,
        }
        for i in range(max_iter):
            optimizer.zero_grad()
            h_val = self.model.h_func(s)
            if h_val.item() < 0:
                self._step.update(iterations=i + 1, stop="h_negative", h=h_val.item())
                return False
            X_hat = self.model(self.X)
            score = self.log_mse_loss(X_hat, self.X)
            l1_reg = self.model.fc1_l1_reg_adaptive(lambda_matrix=lambda_matrix)
            obj = mu * (score + l1_reg) + h_val
            obj.backward()
            optimizer.step()
            if lr_decay and (i + 1) % 1000 == 0:
                scheduler.step()
            if trace_every > 0 and (i % trace_every == 0 or i == max_iter - 1):
                with torch.no_grad():
                    W = self.model.fc1_to_adj()
                    absolute = np.abs(W)
                    self.trace.append(
                        {
                            "outer_step": self._outer,
                            "attempt": self._attempt,
                            "iteration": i,
                            "mu": mu,
                            "s": s,
                            "objective": float(obj.item()),
                            "score": float(score.item()),
                            "l1_penalty": float(l1_reg.item()),
                            "h": float(h_val.item()),
                            "w_l1": float(absolute.sum()),
                            "w_frobenius": float(np.sqrt((absolute**2).sum())),
                            "w_max": float(absolute.max()),
                            "w_above_1e-6": int((absolute > 1e-6).sum()),
                        }
                    )
            if i % self.checkpoint == 0 or i == max_iter - 1:
                obj_new = obj.item()
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    self._step.update(
                        iterations=i + 1,
                        stop="tolerance",
                        relative_change=float(np.abs((obj_prev - obj_new) / obj_prev)),
                    )
                    obj_prev = obj_new
                    break
                self._step.update(relative_change=float(np.abs((obj_prev - obj_new) / obj_prev)))
                obj_prev = obj_new
        with torch.no_grad():
            self._step.update(
                objective=float(obj.item()),
                score=float(score.item()),
                l1_penalty=float(l1_reg.item()),
                h=float(h_val.item()),
            )
        return True

    def fit(
        self,
        X: np.ndarray,
        Lambda: np.ndarray,
        lambda2: float,
        T: int,
        warm_iter: int,
        max_iter: int,
        lr: float,
        mu_init: float = 0.1,
        mu_factor: float = 0.1,
        s: float = 1.0,
        checkpoint: int = 1000,
        trace_every: int = 0,
    ) -> np.ndarray:
        """Follow the central path with priors active from the first step.

        Returns the raw continuous W (no threshold). ``trace_every`` only adds
        the convergence trace (self.trace); 0 disables it and leaves the fit
        exactly as it was.
        """
        torch.set_default_dtype(self.dtype)
        self.X = torch.from_numpy(np.asarray(X)).type(self.dtype)
        lambda_tensor = torch.from_numpy(np.asarray(Lambda, dtype=float)).type(self.dtype)
        self.checkpoint = checkpoint
        mu = mu_init
        s_list = s if isinstance(s, list) else T * [s]
        if len(s_list) < T:
            s_list = s_list + (T - len(s_list)) * [s_list[-1]]
        for i in range(int(T)):
            success, s_cur = False, s_list[i]
            inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)
            model_copy = copy.deepcopy(self.model)
            lr_decay = False
            attempts = 0
            while success is False:
                self._outer, self._attempt = i, attempts + 1
                success = self.minimize(
                    inner_iter,
                    lr,
                    lambda2,
                    mu,
                    s_cur,
                    lambda_matrix=lambda_tensor,
                    lr_decay=lr_decay,
                    trace_every=trace_every,
                )
                attempts += 1
                self.history.append(
                    {
                        "outer_step": i,
                        "attempt": attempts,
                        "last_step": bool(i == int(T) - 1),
                        "success": bool(success),
                        **self._step,
                    }
                )
                if success is False:
                    self.retries += 1
                    self.model.load_state_dict(model_copy.state_dict().copy())
                    lr *= 0.5
                    lr_decay = True
                    if lr < 1e-10:
                        break
                    s_cur = 1
            mu *= mu_factor
        return self.model.fc1_to_adj()
