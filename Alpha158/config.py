from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)


@dataclass(frozen=True)
class Config:
    # Data and experiment periods
    provider_uri: str = os.environ.get("QLIB_DATA", "~/.qlib/qlib_data/cn_data").strip()
    logging_level: str = os.environ.get("QLIB_LOGGING_LEVEL", "INFO").strip().upper()
    market: str = "csi500"
    benchmark: str = "market"
    train_start: str = "2010-01-01"
    train_end: str = "2019-12-31"
    valid_start: str = "2020-01-01"
    valid_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2026-08-01"
    qlib_kernels: int = 4

    # Label and LightGBM
    label_expr: str = "Ref($close, -6)/Ref($close, -1) - 1"
    label_name: str = "LABEL0"
    loss: str = "mse"
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50
    learning_rate: float = 0.2
    colsample_bytree: float = 0.8879
    subsample: float = 0.8789
    lambda_l1: float = 205.6999
    lambda_l2: float = 580.9768
    max_depth: int = 8
    num_leaves: int = 210
    num_threads: int = 20

    # TopkDropout backtest
    topk: int = 50
    n_drop: int = 5
    rebalance_interval: int = 5
    limit_status_enabled: bool = True
    account: float = 100_000_000
    deal_price: str = "close"
    open_cost: float = 0.0005
    close_cost: float = 0.0015
    min_cost: float = 5.0

    output_dir: str = "outputs/alpha158_lightgbm"


CONFIG = Config()
