from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import qlib
from qlib.constant import REG_CN
from qlib.contrib.evaluate_alpha import FUTURE_5D_RETURN_LABEL_EXPR
from qlib.contrib.data.handler import Alpha158
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler, DataHandlerLP

from .config import Split


LABEL_EXPR = FUTURE_5D_RETURN_LABEL_EXPR


def validate_provider_path(path: Path) -> None:
    missing: list[str] = []
    calendars_dir = path / "calendars"
    instruments_dir = path / "instruments"
    features_dir = path / "features"
    if not calendars_dir.is_dir() or not any(calendars_dir.glob("*.txt")):
        missing.append("calendars/*.txt")
    if not instruments_dir.is_dir() or not any(instruments_dir.glob("*.txt")):
        missing.append("instruments/*.txt")
    if not features_dir.is_dir():
        missing.append("features/")
    if missing:
        raise FileNotFoundError(
            f"Qlib provider is incomplete: {path}. Missing {', '.join(missing)}. "
            "Set QLIB_DATA to a built Qlib binary data directory, not the raw download/cache directory."
        )


def init_qlib(provider_uri: str, kernels: int = 0) -> Path:
    path = Path(provider_uri).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Qlib data not found: {path}")
    validate_provider_path(path)
    kwargs = {"provider_uri": str(path), "region": REG_CN}
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


def build_dataset(
    split: Split,
    market: str,
    sample_instruments: int = 0,
    processor_n_jobs: int = -1,
    processor_preset: str = "upstream",
) -> DatasetH:
    instruments: str | list[str] = market
    if sample_instruments > 0:
        instruments = D.list_instruments(D.instruments(market), start_time=split.train[0], end_time=split.test[1], as_list=True)
        instruments = sorted(instruments)[:sample_instruments]

    if processor_preset == "safe":
        infer_processors = [{"class": "ProcessInf", "kwargs": {"n_jobs": processor_n_jobs}}, {"class": "Fillna"}]
    elif processor_preset == "upstream":
        infer_processors = []
    else:
        raise ValueError(f"Unsupported processor preset: {processor_preset}")

    handler = Alpha158(
        instruments=instruments,
        start_time=split.train[0],
        end_time=split.test[1],
        fit_start_time=split.train[0],
        fit_end_time=split.train[1],
        infer_processors=infer_processors,
        learn_processors=[{"class": "DropnaLabel"}],
        label=([LABEL_EXPR], ["LABEL0"]),
    )
    return DatasetH(handler=handler, segments=split.as_dict())


def cache_key(args: argparse.Namespace, split: Split) -> str:
    payload = {
        "provider_uri": str(Path(args.provider_uri).expanduser().resolve()),
        "market": args.market,
        "split": split.as_dict(),
        "sample_instruments": args.sample_instruments,
        "processor_preset": args.processor_preset,
        "label": LABEL_EXPR,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def load_frames(args: argparse.Namespace, split: Split, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache_dir = out_dir / "cache"
    key = cache_key(args, split)
    paths = {name: cache_dir / f"{name}_{key}.pkl" for name in split.as_dict()}
    if args.cache_data and all(path.exists() for path in paths.values()):
        logger.info(f"[data] cache hit: {cache_dir} ({key})")
        return tuple(pd.read_pickle(paths[name]) for name in ("train", "valid", "test"))

    logger.info(
        f"[data] cache miss; building Alpha158 handler and fitting qlib processors "
        f"(processor_preset={args.processor_preset}, processor_n_jobs={args.processor_n_jobs})"
    )
    dataset = build_dataset(split, args.market, args.sample_instruments, args.processor_n_jobs, args.processor_preset)
    frames = []
    for name in ("train", "valid", "test"):
        logger.info(f"[data] preparing {name} segment {split.as_dict()[name]}")
        frames.append(dataset.prepare(name, col_set=DataHandler.CS_ALL, data_key=DataHandlerLP.DK_L).astype(np.float32))
    frames = tuple(frames)
    if args.cache_data:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in zip(("train", "valid", "test"), frames):
            logger.info(f"[data] writing {name} cache: {paths[name]}")
            frame.to_pickle(paths[name])
    return frames


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "LABEL0" not in df.columns:
        raise RuntimeError("LABEL0 is missing from Alpha158 data.")
    x = df.drop(columns=["LABEL0"])
    y = df["LABEL0"]
    mask = y.notna()
    return x.loc[mask], y.loc[mask]
