from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_LABEL_EXPR, Split


MAX_ABS_VALUE = 1e20


def load_dotenv(env_path: Path | None = None) -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env" if env_path is None else env_path
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def validate_provider_path(path: Path) -> None:
    missing: list[str] = []
    if not (path / "calendars").is_dir() or not any((path / "calendars").glob("*.txt")):
        missing.append("calendars/*.txt")
    if not (path / "instruments").is_dir() or not any((path / "instruments").glob("*.txt")):
        missing.append("instruments/*.txt")
    if not (path / "features").is_dir():
        missing.append("features/")
    if missing:
        raise FileNotFoundError(f"Qlib provider is incomplete: {path}. Missing {', '.join(missing)}")


def init_qlib(provider_uri: str | os.PathLike[str], kernels: int = 0) -> Path:
    load_dotenv()
    import qlib
    from qlib.constant import REG_CN

    path = Path(provider_uri).expanduser().resolve()
    validate_provider_path(path)
    kwargs: dict[str, Any] = {"provider_uri": str(path), "region": REG_CN}
    if kernels > 0:
        kwargs["kernels"] = kernels
    qlib.init(**kwargs)
    return path


def previous_trade_day(calendar: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp:
    pos = calendar.searchsorted(date, side="left") - 1
    return calendar[max(pos, 0)]


def first_trade_day_on_or_after(calendar: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp:
    pos = calendar.searchsorted(date, side="left")
    return calendar[min(pos, len(calendar) - 1)]


def auto_split(args: argparse.Namespace) -> Split:
    from qlib.data import D

    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    calendar = calendar[calendar >= pd.Timestamp(args.train_start)]
    if len(calendar) < 252 * (args.valid_years + args.test_years + 1):
        raise RuntimeError("Not enough calendar data after train-start for train/valid/test split.")
    test_end = calendar[-1]
    test_start = first_trade_day_on_or_after(calendar, test_end - pd.DateOffset(years=args.test_years))
    valid_start = first_trade_day_on_or_after(calendar, test_start - pd.DateOffset(years=args.valid_years))
    train_start = first_trade_day_on_or_after(calendar, pd.Timestamp(args.train_start))
    if args.train_years > 0:
        train_start = max(train_start, first_trade_day_on_or_after(calendar, valid_start - pd.DateOffset(years=args.train_years)))
    train_end = previous_trade_day(calendar, valid_start)
    valid_end = previous_trade_day(calendar, test_start)
    if not train_start < train_end < valid_start < valid_end < test_start < test_end:
        raise RuntimeError(f"Invalid split: train={train_start, train_end}, valid={valid_start, valid_end}, test={test_start, test_end}")
    return Split(
        (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
        (valid_start.strftime("%Y-%m-%d"), valid_end.strftime("%Y-%m-%d")),
        (test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")),
    )


class QlibAlphaCalculator:
    def __init__(
        self,
        instruments: str | Sequence[str],
        start_time: str,
        end_time: str,
        *,
        target: str = DEFAULT_LABEL_EXPR,
        freq: str = "day",
        normalize_alpha: bool = True,
        sample_instruments: int = 0,
    ) -> None:
        self.instruments = self._resolve_instruments(instruments, start_time, end_time, sample_instruments)
        self.start_time = start_time
        self.end_time = end_time
        self.target_expr = target
        self.freq = freq
        self.normalize_alpha = normalize_alpha
        self._alpha_cache: dict[str, pd.Series] = {}
        self._target_cache: pd.Series | None = None

    @staticmethod
    def _resolve_instruments(instruments: str | Sequence[str], start_time: str, end_time: str, sample_instruments: int):
        from qlib.data import D

        if isinstance(instruments, str) and "," not in instruments:
            qlib_instruments = D.instruments(instruments)
            if sample_instruments > 0:
                inst_list = D.list_instruments(qlib_instruments, start_time=start_time, end_time=end_time, freq="day", as_list=True)
                return sorted(inst_list)[:sample_instruments]
            return qlib_instruments
        if isinstance(instruments, str):
            return [item.strip() for item in instruments.split(",") if item.strip()]
        return list(instruments)

    @property
    def target(self) -> pd.Series:
        if self._target_cache is None:
            self._target_cache = self._load_series(self.target_expr, "label")
        return self._target_cache

    def evaluate_alpha(self, expr: Any) -> pd.Series:
        qlib_expr = str(expr)
        if qlib_expr not in self._alpha_cache:
            series = self._load_series(qlib_expr, "score")
            if self.normalize_alpha:
                series = self._normalize_by_day(series)
            self._alpha_cache[qlib_expr] = series
        return self._alpha_cache[qlib_expr]

    def make_ensemble_alpha(self, exprs: Sequence[Any], weights: Sequence[float]) -> pd.Series:
        if len(exprs) != len(weights):
            raise ValueError(f"exprs and weights length mismatch: {len(exprs)} != {len(weights)}")
        if len(exprs) == 0:
            raise ValueError("At least one expression is required")
        values = [self.evaluate_alpha(expr).mul(float(weight)) for expr, weight in zip(exprs, weights)]
        result = values[0].copy()
        for value in values[1:]:
            result = result.add(value, fill_value=0.0)
        result.name = "score"
        return result

    def calc_single_IC_ret(self, expr: Any) -> float:
        return self._calc_ic_pair(self.evaluate_alpha(expr), self.target)[0]

    def calc_single_rIC_ret(self, expr: Any) -> float:
        return self._calc_ic_pair(self.evaluate_alpha(expr), self.target)[1]

    def calc_single_all_ret(self, expr: Any) -> tuple[float, float]:
        return self._calc_ic_pair(self.evaluate_alpha(expr), self.target)

    def calc_mutual_IC(self, expr1: Any, expr2: Any) -> float:
        return self._calc_ic_pair(self.evaluate_alpha(expr1), self.evaluate_alpha(expr2))[0]

    def calc_pool_IC_ret(self, exprs: Sequence[Any], weights: Sequence[float]) -> float:
        return self._calc_ic_pair(self.make_ensemble_alpha(exprs, weights), self.target)[0]

    def calc_pool_rIC_ret(self, exprs: Sequence[Any], weights: Sequence[float]) -> float:
        return self._calc_ic_pair(self.make_ensemble_alpha(exprs, weights), self.target)[1]

    def calc_pool_all_ret(self, exprs: Sequence[Any], weights: Sequence[float]) -> tuple[float, float]:
        return self._calc_ic_pair(self.make_ensemble_alpha(exprs, weights), self.target)

    def calc_summary(self, score: pd.Series, label: pd.Series | None = None) -> pd.Series:
        from qlib.contrib.evaluate_alpha import get_ic_summary

        return get_ic_summary(self._calc_ic_frame(score, self.target if label is None else label))

    def _load_series(self, expr: str, column_name: str) -> pd.Series:
        from qlib.data import D

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="overflow encountered", category=RuntimeWarning)
            df = D.features(
                self.instruments,
                [expr],
                start_time=self.start_time,
                end_time=self.end_time,
                freq=self.freq,
                disk_cache=0,
            )
        if df.empty:
            raise RuntimeError(f"No qlib data returned for expression: {expr}")
        df = df.rename(columns={df.columns[0]: column_name})
        series = self._sanitize_series(df[column_name].sort_index())
        series.name = column_name
        return series

    @staticmethod
    def _normalize_by_day(series: pd.Series) -> pd.Series:
        from qlib.data.dataset.processor import CSZScoreNorm, Fillna

        df = QlibAlphaCalculator._sanitize_series(series).to_frame("score")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="overflow encountered", category=RuntimeWarning)
            df = CSZScoreNorm(fields_group=None)(df)
            df = Fillna(fill_value=0)(df)
        result = df["score"]
        result.name = series.name
        return result

    @staticmethod
    def _pred_label(score: pd.Series, label: pd.Series) -> pd.DataFrame:
        score = QlibAlphaCalculator._sanitize_series(score)
        label = QlibAlphaCalculator._sanitize_series(label)
        return pd.concat({"score": score, "label": label.reindex(score.index)}, axis=1)

    def _calc_ic_frame(self, score: pd.Series, label: pd.Series) -> pd.DataFrame:
        from qlib.contrib.evaluate_alpha import get_score_ic

        return get_score_ic(self._pred_label(score, label))

    def _calc_ic_pair(self, score: pd.Series, label: pd.Series) -> tuple[float, float]:
        from qlib.contrib.evaluate_alpha import get_ic_summary

        summary = get_ic_summary(self._calc_ic_frame(score, label))
        return float(summary["IC"]), float(summary["Rank IC"])

    @staticmethod
    def _sanitize_series(series: pd.Series) -> pd.Series:
        result = series.replace([np.inf, -np.inf], np.nan)
        result = result.mask(result.abs() > MAX_ABS_VALUE)
        return result


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
