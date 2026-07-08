from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class PoolUpdate:
    expr: Any
    reward: float
    accepted: bool
    ic: float | None = None


class MseAlphaPool:
    def __init__(
        self,
        capacity: int,
        calculator,
        *,
        ic_lower_bound: float | None = None,
        l1_alpha: float = 5e-3,
        mutual_ic_threshold: float = 0.99,
        optimize_max_steps: int = 300,
        optimize_tolerance: int = 50,
    ) -> None:
        self.capacity = capacity
        self.calculator = calculator
        self.ic_lower_bound = -1.0 if ic_lower_bound is None else ic_lower_bound
        self.l1_alpha = l1_alpha
        self.mutual_ic_threshold = mutual_ic_threshold
        self.optimize_max_steps = optimize_max_steps
        self.optimize_tolerance = optimize_tolerance
        self.exprs: list[Any] = []
        self.single_ics = np.zeros(capacity + 1, dtype=np.float64)
        self.weights = np.zeros(capacity + 1, dtype=np.float64)
        self.mutual_ics = np.identity(capacity + 1, dtype=np.float64)
        self.best_ic_ret = -1.0
        self.best_obj = -1.0
        self.eval_cnt = 0
        self.failure_cache: set[str] = set()
        self.history: list[PoolUpdate] = []

    @property
    def size(self) -> int:
        return len(self.exprs)

    def to_json_dict(self) -> dict:
        return {
            "exprs": [str(expr) for expr in self.exprs],
            "weights": [float(w) for w in self.weights[: self.size]],
            "single_ics": [float(x) for x in self.single_ics[: self.size]],
            "best_ic_ret": float(self.best_ic_ret),
            "best_obj": float(self.best_obj),
            "eval_cnt": self.eval_cnt,
        }

    def force_load_exprs(self, exprs: Sequence[Any], weights: Sequence[float] | None = None) -> None:
        for expr in exprs:
            ic, mutual = self._calc_ics(expr, threshold=None)
            if ic is None or mutual is None or not np.isfinite(ic):
                continue
            self._add(expr, ic, mutual)
        if weights is not None:
            if len(weights) != self.size:
                raise ValueError(f"Invalid weights length: {len(weights)} != {self.size}")
            self.weights[: self.size] = np.asarray(weights, dtype=np.float64)
        elif self.size:
            self.weights[: self.size] = self.optimize()
        self._update_best()

    def try_new_expr(self, expr: Any) -> float:
        key = str(expr)
        if key in self.failure_cache:
            return self.best_obj
        try:
            ic, mutual = self._calc_ics(expr, threshold=self.mutual_ic_threshold)
        except Exception:
            self.failure_cache.add(key)
            self.history.append(PoolUpdate(expr, 0.0, False))
            return 0.0
        if ic is None or mutual is None or np.isnan(ic) or not np.all(np.isfinite(mutual)):
            self.failure_cache.add(key)
            self.history.append(PoolUpdate(expr, 0.0, False, ic))
            return 0.0
        if self.size > 0 and ic < self.ic_lower_bound:
            self.failure_cache.add(key)
            self.history.append(PoolUpdate(expr, self.best_obj, False, ic))
            return self.best_obj

        self.eval_cnt += 1
        old_obj = self.best_obj
        self._add(expr, ic, mutual)
        self.weights[: self.size] = self.optimize()
        if self.size > self.capacity:
            worst = int(np.argmin(np.abs(self.weights[: self.size])))
            removed = self.exprs.pop(worst)
            self._delete_index(worst)
            if removed is expr:
                self.failure_cache.add(key)
                self.history.append(PoolUpdate(expr, self.best_obj, False, ic))
                return self.best_obj
        self.failure_cache.clear()
        self._update_best()
        reward = self.best_obj if math.isfinite(self.best_obj) else 0.0
        self.history.append(PoolUpdate(expr, reward, self.best_obj > old_obj, ic))
        return reward

    def evaluate_ensemble(self) -> float:
        if self.size == 0:
            return 0.0
        return self.calculator.calc_pool_IC_ret(self.exprs, self.weights[: self.size])

    def test_ensemble(self, calculator) -> tuple[float, float]:
        if self.size == 0:
            return 0.0, 0.0
        return calculator.calc_pool_all_ret(self.exprs, self.weights[: self.size])

    def optimize(self) -> np.ndarray:
        if self.size == 0:
            return np.array([], dtype=np.float64)
        if self.l1_alpha == 0 or self.size <= 2:
            return self._optimize_lstsq()
        try:
            import torch

            ics_ret = torch.tensor(self.single_ics[: self.size], dtype=torch.float32)
            ics_mut = torch.tensor(self.mutual_ics[: self.size, : self.size], dtype=torch.float32)
            weights = torch.tensor(self.weights[: self.size], dtype=torch.float32, requires_grad=True)
            optimizer = torch.optim.Adam([weights], lr=5e-4)
            best = weights.detach().clone()
            best_loss = float("inf")
            stale = 0
            for _ in range(self.optimize_max_steps):
                ret_ic_sum = (weights * ics_ret).sum()
                mut_ic_sum = (torch.outer(weights, weights) * ics_mut).sum()
                loss_ic = mut_ic_sum - 2 * ret_ic_sum + 1
                loss = loss_ic + self.l1_alpha * torch.norm(weights, p=1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                current = float(loss_ic.detach().cpu())
                if best_loss - current > 1e-6:
                    best_loss = current
                    best = weights.detach().clone()
                    stale = 0
                else:
                    stale += 1
                if stale >= self.optimize_tolerance:
                    break
            return best.cpu().numpy()
        except ImportError:
            return self._optimize_lstsq()

    def _optimize_lstsq(self) -> np.ndarray:
        try:
            return np.linalg.lstsq(self.mutual_ics[: self.size, : self.size], self.single_ics[: self.size], rcond=None)[0]
        except (np.linalg.LinAlgError, ValueError):
            return self.weights[: self.size]

    def _calc_ics(self, expr: Any, threshold: float | None) -> tuple[float | None, list[float] | None]:
        ic = self.calculator.calc_single_IC_ret(expr)
        mutual: list[float] = []
        for existing in self.exprs:
            mic = self.calculator.calc_mutual_IC(expr, existing)
            if threshold is not None and mic > threshold:
                return ic, None
            mutual.append(mic)
        return ic, mutual

    def _add(self, expr: Any, ic: float, mutual: list[float]) -> None:
        idx = self.size
        self.exprs.append(expr)
        self.single_ics[idx] = ic
        for i, mic in enumerate(mutual):
            self.mutual_ics[i, idx] = self.mutual_ics[idx, i] = mic
        self.weights[idx] = max(ic, 0.01) if idx == 0 else float(np.mean(self.weights[:idx]))

    def _delete_index(self, idx: int) -> None:
        size = self.size + 1
        self.single_ics[idx : size - 1] = self.single_ics[idx + 1 : size]
        self.weights[idx : size - 1] = self.weights[idx + 1 : size]
        self.mutual_ics[idx : size - 1, :] = self.mutual_ics[idx + 1 : size, :]
        self.mutual_ics[:, idx : size - 1] = self.mutual_ics[:, idx + 1 : size]

    def _update_best(self) -> None:
        ic = self.evaluate_ensemble()
        obj = ic
        if obj > self.best_obj:
            self.best_obj = obj
            self.best_ic_ret = ic
