from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from dotenv import load_dotenv


load_dotenv()


ALL_MODELS = [
    "linear",
    "xgboost",
    "lightgbm",
    "mlp",
    "gru",
    "tra",
    "lstm",
    "transformer",
]


@dataclass
class Split:
    train: tuple[str, str]
    valid: tuple[str, str]
    test: tuple[str, str]

    def as_dict(self) -> dict[str, tuple[str, str]]:
        return {"train": self.train, "valid": self.valid, "test": self.test}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Alpha158 model comparison and generate reports.")
    parser.add_argument("--provider-uri", default=qlib_data_dir())
    parser.add_argument("--market", default="csi500")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--models", nargs="+", default=["all"], help="all or a subset of model names")
    parser.add_argument("--train-start", default="2008-01-01")
    parser.add_argument("--train-years", type=int, default=10)
    parser.add_argument("--valid-years", type=int, default=1)
    parser.add_argument("--test-years", type=int, default=4)
    parser.add_argument("--topk-ratio", type=float, default=0.2)
    parser.add_argument("--step-len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--tree-n-estimators", type=int, default=800)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument(
        "--lightgbm-device",
        choices=["cpu"],
        default="cpu",
        help="LightGBM backend. This project runs LightGBM on CPU to avoid CUDA/OpenCL runtime requirements.",
    )
    parser.add_argument(
        "--xgboost-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="XGBoost backend. auto uses CUDA only when PyTorch reports an available CUDA device.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qlib-kernels", type=int, default=8, help="Qlib feature loading workers; 0 means qlib default")
    parser.add_argument("--processor-n-jobs", type=int, default=8, help="Joblib workers for Alpha158 processors; -1 means all CPUs")
    parser.add_argument(
        "--processor-preset",
        choices=["upstream", "safe"],
        default="upstream",
        help="upstream skips slow ProcessInf like qlib Alpha158 tree configs; safe keeps ProcessInf+Fillna",
    )
    parser.add_argument(
        "--sequence-feature-preset",
        choices=["alpha20", "all"],
        default="alpha20",
        help="alpha20 follows upstream GRU/LSTM/TRA Alpha158 configs and keeps 20 time-series features",
    )
    parser.add_argument("--cache-data", action="store_true", help="Cache processed Alpha158 frames under output-dir/cache")
    parser.add_argument("--sample-instruments", type=int, default=0, help="Debug mode: keep first N instruments only")
    parser.add_argument("--fast-dev", action="store_true", help="Short run for smoke tests")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow PyTorch models to run on CPU")
    return parser.parse_args()


def require_optional(package: str, import_name: str | None = None):
    try:
        return __import__(import_name or package)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing dependency `{package}`. Activate `qlib-reloaded` or run "
            "`python -m pip install -r requirements.txt` in that environment."
        ) from exc


def selected_models(names: Iterable[str]) -> list[str]:
    requested = [x.lower() for x in names]
    if "all" in requested:
        return ALL_MODELS
    unknown = sorted(set(requested) - set(ALL_MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Valid names: {ALL_MODELS}")
    return requested


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def qlib_data_dir() -> str:
    return os.environ.get("QLIB_DATA", "~/.qlib/qlib_data/cn_data")
