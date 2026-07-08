# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import pandas as pd

from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.evaluate_portfolio import get_normal_ic, get_rank_ic


FUTURE_5D_RETURN_LABEL_EXPR = "Ref($close, -6)/Ref($close, -1) - 1"


def get_score_ic(pred_label: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily IC and Rank IC for a score/label multi-index frame."""
    concat_data = pred_label[["score", "label"]].copy()
    concat_data.dropna(axis=0, how="any", inplace=True)

    def _normal_ic(day: pd.DataFrame) -> float:
        return get_normal_ic(day["score"], day["label"])

    def _rank_ic(day: pd.DataFrame) -> float:
        return get_rank_ic(day["score"], day["label"])

    ic = concat_data.groupby(level="datetime", group_keys=False).apply(_normal_ic)
    rank_ic = concat_data.groupby(level="datetime", group_keys=False).apply(_rank_ic)
    return pd.DataFrame({"IC": ic, "Rank IC": rank_ic}).replace([np.inf, -np.inf], np.nan).dropna()


def get_topk_return(pred_label: pd.DataFrame, topk_ratio: float) -> pd.DataFrame:
    """Calculate label-based top-k long, benchmark, and excess returns."""
    concat_data = pred_label[["score", "label"]].copy()
    concat_data.dropna(axis=0, how="any", inplace=True)

    def one_day(day: pd.DataFrame) -> pd.Series:
        n = max(1, int(len(day) * topk_ratio))
        long_ret = day.nlargest(n, "score")["label"].mean()
        bench_ret = day["label"].mean()
        return pd.Series({"long": long_ret, "benchmark": bench_ret, "excess": long_ret - bench_ret})

    return concat_data.groupby(level="datetime").apply(one_day)


def get_return_risk(return_series: pd.Series, freq: str = "day") -> pd.Series:
    """Return qlib risk metrics as a flat Series."""
    risk = risk_analysis(return_series.dropna(), freq=freq, mode="sum")["risk"]
    return pd.Series(
        {
            "IR (SHR*)": risk["information_ratio"],
            "ARR (%)": risk["annualized_return"] * 100.0,
            "MDD (%)": risk["max_drawdown"] * 100.0,
        }
    )


def get_ic_summary(ic_frame: pd.DataFrame) -> pd.Series:
    """Summarize qlib IC frame into mean and information-ratio style values."""
    ic_std = ic_frame["IC"].std(ddof=1)
    rank_ic_std = ic_frame["Rank IC"].std(ddof=1)
    return pd.Series(
        {
            "IC": ic_frame["IC"].mean(),
            "ICIR": ic_frame["IC"].mean() / ic_std if ic_std else np.nan,
            "Rank IC": ic_frame["Rank IC"].mean(),
            "Rank ICIR": ic_frame["Rank IC"].mean() / rank_ic_std if rank_ic_std else np.nan,
        }
    )
