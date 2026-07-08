from __future__ import annotations

import pandas as pd

from qlib.contrib.evaluate_alpha import get_ic_summary, get_return_risk, get_score_ic, get_topk_return


def build_pred_label(pred: pd.Series, label: pd.Series) -> pd.DataFrame:
    return pd.concat({"score": pred, "label": label.reindex(pred.index)}, axis=1)


def evaluate_model(name: str, pred: pd.Series, test_label: pd.Series, topk_ratio: float) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    pred_label = build_pred_label(pred, test_label)
    ic_frame = get_score_ic(pred_label)
    returns = get_topk_return(pred_label, topk_ratio)
    summary = pd.concat([get_ic_summary(ic_frame), get_return_risk(returns["excess"])])
    row = {"model": name, **summary.to_dict()}
    return row, returns, ic_frame
