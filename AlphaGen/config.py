from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from dotenv import load_dotenv


load_dotenv()

DEFAULT_LABEL_EXPR = "Ref($close, -6)/Ref($close, -1) - 1"
DEFAULT_MARKET = "csi500"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_TRAIN_START = "2008-01-01"
DEFAULT_TRAIN_YEARS = 10
DEFAULT_VALID_YEARS = 1
DEFAULT_TEST_YEARS = 4
DEFAULT_QLIB_KERNELS = 8

DEFAULT_POOL_CAPACITY = 20
DEFAULT_MAX_EXPR_LENGTH = 15
DEFAULT_STEPS = 5_000
DEFAULT_EPISODE_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_NUM_LAYERS = 2
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_L1_ALPHA = 5e-3
DEFAULT_PPO_EPOCHS = 4
DEFAULT_PPO_CLIP = 0.2
DEFAULT_MINE_YEARS = 2
DEFAULT_POOL_OPT_STEPS = 300
DEFAULT_POOL_OPT_TOLERANCE = 50


FEATURES = ["open", "close", "high", "low", "volume", "vwap"]
DELTA_TIMES = [1, 5, 10, 20, 40]
CONSTANTS = [-30.0, -10.0, -5.0, -2.0, -1.0, -0.5, -0.01, 0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]


@dataclass
class Split:
    train: tuple[str, str]
    valid: tuple[str, str]
    test: tuple[str, str]

    def as_dict(self) -> dict[str, tuple[str, str]]:
        return {"train": self.train, "valid": self.valid, "test": self.test}


def qlib_data_dir() -> str:
    return os.environ.get("QLIB_DATA", "~/.qlib/qlib_data/cn_data")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a qlib-native AlphaGen implementation without using the upstream AlphaGen source."
    )
    parser.add_argument("--provider-uri", default=qlib_data_dir())
    parser.add_argument("--market", default=DEFAULT_MARKET)
    parser.add_argument("--instruments", default=None, help="Optional comma-separated instrument list. Overrides --market.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-start", default=DEFAULT_TRAIN_START)
    parser.add_argument("--train-years", type=int, default=DEFAULT_TRAIN_YEARS)
    parser.add_argument("--valid-years", type=int, default=DEFAULT_VALID_YEARS)
    parser.add_argument("--test-years", type=int, default=DEFAULT_TEST_YEARS)
    parser.add_argument("--freq", default="day")
    parser.add_argument("--target", default=DEFAULT_LABEL_EXPR)
    parser.add_argument("--sample-instruments", type=int, default=0)
    parser.add_argument("--qlib-kernels", type=int, default=DEFAULT_QLIB_KERNELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mine-years",
        type=int,
        default=DEFAULT_MINE_YEARS,
        help="Use the tail N years of the train segment for RL reward. 0 uses the full train segment.",
    )

    parser.add_argument("--pool-capacity", type=int, default=DEFAULT_POOL_CAPACITY)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Number of generated episodes.")
    parser.add_argument("--episode-batch-size", type=int, default=DEFAULT_EPISODE_BATCH_SIZE)
    parser.add_argument("--max-expr-length", type=int, default=DEFAULT_MAX_EXPR_LENGTH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--entropy-coef", type=float, default=DEFAULT_ENTROPY_COEF)
    parser.add_argument("--ppo-epochs", type=int, default=DEFAULT_PPO_EPOCHS)
    parser.add_argument("--ppo-clip", type=float, default=DEFAULT_PPO_CLIP)
    parser.add_argument("--l1-alpha", type=float, default=DEFAULT_L1_ALPHA)
    parser.add_argument("--pool-opt-steps", type=int, default=DEFAULT_POOL_OPT_STEPS)
    parser.add_argument("--pool-opt-tolerance", type=int, default=DEFAULT_POOL_OPT_TOLERANCE)
    parser.add_argument("--ic-lower-bound", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-csrank", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--print-expr", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)

    parser.add_argument("--expr", action="append", help="Evaluate existing expression(s) instead of running generation.")
    parser.add_argument("--pool-json", help="Evaluate an existing generated pool instead of running generation.")
    parser.add_argument("--weights", help="Comma-separated weights for --expr values.")
    parser.add_argument("--start", help="Evaluation start date for --expr/--pool-json mode.")
    parser.add_argument("--end", help="Evaluation end date for --expr/--pool-json mode.")
    return parser.parse_args(list(argv) if argv is not None else None)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
