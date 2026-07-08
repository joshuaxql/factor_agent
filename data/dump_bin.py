"""Dump cached GM daily CSV data into qlib's local binary provider format.

The downloader writes intermediate files under ``<QLIB_DATA>/cache``.  This
module converts those files into the provider layout consumed by ``qlib.init``::

    calendars/day.txt
    instruments/all.txt
    instruments/csi300.txt
    features/<instrument>/<field>.day.bin

Feature ``.bin`` files follow qlib's file storage convention: the first
``float32`` is the start index in ``calendars/day.txt`` and the remaining
``float32`` values are a contiguous calendar-aligned series.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from data.config import CACHE_DIR, CONSTITUENTS_DIR, DAILY_DIR, QLIB_DATA


INDEX_MARKET_NAMES: dict[str, str] = {
    "sh000300": "csi300",
    "sh000905": "csi500",
    "sh000906": "csi800",
    "sh000852": "csi1000",
}

QLIB_OUTPUT_DIRS = ("calendars", "instruments", "features")


def _date_str(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _filter_dates(dates: Iterable[str], start: str | None, end: str | None) -> list[str]:
    start_s = _date_str(start) if start else None
    end_s = _date_str(end) if end else None
    result = []
    for date in sorted({_date_str(d) for d in dates}):
        if start_s and date < start_s:
            continue
        if end_s and date > end_s:
            continue
        result.append(date)
    return result


def _load_calendar(cache_dir: Path, daily_files: list[Path], start: str | None, end: str | None) -> list[str]:
    trading_dates_path = cache_dir / "trading_dates.csv"
    if trading_dates_path.exists():
        df = pd.read_csv(trading_dates_path)
        if "date" not in df.columns:
            raise ValueError(f"{trading_dates_path} must contain a date column")
        dates = df["date"].dropna().astype(str).tolist()
    else:
        dates = [p.stem for p in daily_files]
        logger.warning(f"{trading_dates_path} not found; using daily CSV filenames as calendar")

    calendar = _filter_dates(dates, start, end)
    if not calendar:
        raise ValueError("No calendar dates found for the requested range")
    return calendar


def _daily_files(daily_dir: Path, start: str | None, end: str | None) -> list[Path]:
    files = sorted(daily_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No daily CSV files found in {daily_dir}")

    wanted_dates = set(_filter_dates((p.stem for p in files), start, end))
    selected = [p for p in files if p.stem in wanted_dates]
    if not selected:
        raise FileNotFoundError(f"No daily CSV files found in {daily_dir} for the requested range")
    return selected


def _read_daily_frame(
    files: list[Path],
    sample_instruments: int = 0,
    include_aliases: bool = False,
) -> pd.DataFrame:
    frames = []
    for path in tqdm(files, desc="读取日线CSV", ncols=80):
        frames.append(pd.read_csv(path, dtype={"symbol": str, "date": str}))

    df = pd.concat(frames, ignore_index=True, sort=False)
    if df.empty:
        raise ValueError("Daily CSV files are empty")
    if "symbol" not in df.columns or "date" not in df.columns:
        raise ValueError("Daily CSV must contain symbol and date columns")

    df["symbol"] = df["symbol"].astype(str).str.lower()
    df["date"] = df["date"].map(_date_str)

    if sample_instruments > 0:
        symbols = sorted(df["symbol"].unique())[:sample_instruments]
        df = df[df["symbol"].isin(symbols)].copy()
        logger.info(f"sample-instruments enabled: keeping {len(symbols)} instruments")

    return _derive_feature_columns(df, include_aliases=include_aliases)


def _derive_feature_columns(df: pd.DataFrame, include_aliases: bool = False) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    for col in df.columns:
        if col not in {"symbol", "date"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if include_aliases:
        if "amount" in df.columns and "money" not in df.columns:
            df["money"] = df["amount"]
        if "turn_rate" in df.columns:
            if "turn" not in df.columns:
                df["turn"] = df["turn_rate"]
            if "turnover" not in df.columns:
                df["turnover"] = df["turn_rate"]
    if "adj_factor" in df.columns and "factor" not in df.columns:
        df["factor"] = df["adj_factor"]

    if "close" in df.columns:
        df["change"] = df.groupby("symbol", sort=False)["close"].pct_change(fill_method=None)

    numeric_cols = [c for c in df.columns if c not in {"symbol", "date"}]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df


def _feature_fields(df: pd.DataFrame, fields: Optional[list[str]]) -> list[str]:
    if fields:
        missing = [f for f in fields if f not in df.columns]
        if missing:
            raise ValueError(f"Requested fields are missing from daily data: {missing}")
        return fields

    fields = []
    for col in df.columns:
        if col in {"symbol", "date"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fields.append(col.lower())
    return sorted(set(fields), key=fields.index)


def _write_calendar(provider_uri: Path, calendar: list[str]) -> None:
    calendars_dir = provider_uri / "calendars"
    calendars_dir.mkdir(parents=True, exist_ok=True)
    (calendars_dir / "day.txt").write_text("\n".join(calendar) + "\n", encoding="utf-8")
    logger.info(f"calendar written: {calendars_dir / 'day.txt'} ({len(calendar)} dates)")


def _write_bin(path: Path, values: np.ndarray, start_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.hstack(([float(start_index)], values.astype(np.float32, copy=False))).astype("<f4", copy=False)
    payload.tofile(path)


def _write_features(provider_uri: Path, df: pd.DataFrame, calendar: list[str], fields: list[str]) -> dict[str, tuple[str, str]]:
    features_dir = provider_uri / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    calendar_index = pd.Index(calendar, name="date")
    spans: dict[str, tuple[str, str]] = {}

    grouped = df.groupby("symbol", sort=True)
    for symbol, group in tqdm(grouped, total=df["symbol"].nunique(), desc="写入features", ncols=80):
        group = group.drop_duplicates("date", keep="last").set_index("date").sort_index()
        spans[symbol] = (str(group.index.min()), str(group.index.max()))
        aligned = group.reindex(calendar_index)

        for field in fields:
            if field not in aligned.columns:
                continue
            values = pd.to_numeric(aligned[field], errors="coerce").to_numpy(dtype=np.float32)
            valid_pos = np.flatnonzero(~np.isnan(values))
            if len(valid_pos) == 0:
                continue
            start_idx = int(valid_pos[0])
            end_idx = int(valid_pos[-1])
            out_path = features_dir / symbol / f"{field.lower()}.day.bin"
            _write_bin(out_path, values[start_idx : end_idx + 1], start_idx)

    logger.info(f"features written: {features_dir} ({len(spans)} instruments, {len(fields)} fields)")
    return spans


def _write_instrument_file(path: Path, spans: dict[str, list[tuple[str, str]]] | dict[str, tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for symbol in sorted(spans):
        symbol_spans = spans[symbol]
        if isinstance(symbol_spans, tuple):
            symbol_spans = [symbol_spans]
        for start, end in symbol_spans:
            lines.append(f"{symbol}\t{_date_str(start)}\t{_date_str(end)}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_all_instruments(provider_uri: Path, spans: dict[str, tuple[str, str]]) -> None:
    instruments_dir = provider_uri / "instruments"
    _write_instrument_file(instruments_dir / "all.txt", spans)
    _write_instrument_file(instruments_dir / "csiall.txt", spans)
    logger.info(f"all instruments written: {len(spans)}")


def _indices_to_spans(indices: list[int], calendar: list[str]) -> list[tuple[str, str]]:
    if not indices:
        return []
    result = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        result.append((calendar[start], calendar[prev]))
        start = prev = idx
    result.append((calendar[start], calendar[prev]))
    return result


def _build_constituent_spans(
    const_dir: Path,
    calendar: list[str],
    allowed_symbols: set[str] | None = None,
) -> dict[str, list[tuple[str, str]]]:
    date_to_idx = {date: idx for idx, date in enumerate(calendar)}
    member_indices: dict[str, list[int]] = defaultdict(list)

    files = sorted(p for p in const_dir.glob("*.csv") if p.stem in date_to_idx)
    for path in tqdm(files, desc=f"读取{const_dir.name}成分", ncols=80, leave=False):
        try:
            df = pd.read_csv(path, dtype={"symbol": str})
        except pd.errors.EmptyDataError:
            logger.debug(f"skip empty constituent file: {path}")
            continue
        if "symbol" not in df.columns:
            continue
        idx = date_to_idx[path.stem]
        for symbol in df["symbol"].dropna().astype(str).str.lower().unique():
            if allowed_symbols is not None and symbol not in allowed_symbols:
                continue
            member_indices[symbol].append(idx)

    return {symbol: _indices_to_spans(sorted(set(indices)), calendar) for symbol, indices in member_indices.items()}


def _write_index_instruments(
    provider_uri: Path,
    constituents_dir: Path,
    calendar: list[str],
    allowed_symbols: set[str],
) -> None:
    instruments_dir = provider_uri / "instruments"
    if not constituents_dir.exists():
        logger.warning(f"constituents directory not found: {constituents_dir}")
        return

    for const_dir in sorted(p for p in constituents_dir.iterdir() if p.is_dir()):
        market = INDEX_MARKET_NAMES.get(const_dir.name.lower(), const_dir.name.lower())
        spans = _build_constituent_spans(const_dir, calendar, allowed_symbols=allowed_symbols)
        if not spans:
            logger.warning(f"{market}: no constituent spans built from {const_dir}")
            continue
        _write_instrument_file(instruments_dir / f"{market}.txt", spans)
        logger.info(f"{market} instruments written: {len(spans)} symbols")


def _clear_existing_provider(provider_uri: Path) -> None:
    resolved = provider_uri.resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError(f"Refuse to clear drive/root directory: {resolved}")
    for name in QLIB_OUTPUT_DIRS:
        target = resolved / name
        if target.exists():
            shutil.rmtree(target)
            logger.info(f"removed existing qlib output: {target}")


def verify_provider(provider_uri: str | Path) -> None:
    provider_uri = _resolve_path(provider_uri)
    missing = []
    if not (provider_uri / "calendars" / "day.txt").is_file():
        missing.append("calendars/day.txt")
    if not (provider_uri / "instruments" / "all.txt").is_file():
        missing.append("instruments/all.txt")
    if not any((provider_uri / "features").glob("*/*.day.bin")):
        missing.append("features/*/*.day.bin")
    if missing:
        raise FileNotFoundError(f"Qlib provider is incomplete: {provider_uri}. Missing {', '.join(missing)}")
    logger.info(f"provider verified: {provider_uri}")


def dump_qlib_data(
    provider_uri: str | Path = QLIB_DATA,
    cache_dir: str | Path = CACHE_DIR,
    daily_dir: str | Path | None = None,
    constituents_dir: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
    fields: Optional[list[str]] = None,
    sample_instruments: int = 0,
    include_aliases: bool = False,
    clear_existing: bool = False,
    verify: bool = True,
) -> None:
    """Convert cached daily CSV files into qlib binary provider files."""
    provider_uri = _resolve_path(provider_uri)
    cache_dir = _resolve_path(cache_dir)
    daily_dir = _resolve_path(daily_dir) if daily_dir is not None else cache_dir / "daily"
    constituents_dir = _resolve_path(constituents_dir) if constituents_dir is not None else cache_dir / "constituents"

    files = _daily_files(daily_dir, start, end)
    calendar = _load_calendar(cache_dir, files, start, end)
    daily_dates = {p.stem for p in files}
    missing_daily_dates = sorted(set(calendar) - daily_dates)
    if missing_daily_dates:
        logger.warning(
            f"{len(missing_daily_dates)} calendar dates have no daily CSV; feature values will be NaN "
            f"(first missing: {missing_daily_dates[0]})"
        )

    if clear_existing:
        _clear_existing_provider(provider_uri)

    df = _read_daily_frame(files, sample_instruments=sample_instruments, include_aliases=include_aliases)
    feature_fields = _feature_fields(df, fields)
    if not feature_fields:
        raise ValueError("No numeric feature fields found to dump")

    provider_uri.mkdir(parents=True, exist_ok=True)
    _write_calendar(provider_uri, calendar)
    spans = _write_features(provider_uri, df, calendar, feature_fields)
    _write_all_instruments(provider_uri, spans)
    _write_index_instruments(provider_uri, constituents_dir, calendar, allowed_symbols=set(spans))

    if verify:
        verify_provider(provider_uri)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将掘金缓存日线 CSV 转换为 qlib bin 格式")
    parser.add_argument("--provider-uri", default=str(QLIB_DATA), help="qlib provider 输出目录，默认读取 QLIB_DATA")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR), help="缓存根目录，默认 <QLIB_DATA>/cache")
    parser.add_argument("--daily-dir", default=None, help="按日 CSV 目录，默认 <cache-dir>/daily")
    parser.add_argument("--constituents-dir", default=None, help="指数成分缓存目录，默认 <cache-dir>/constituents")
    parser.add_argument("--start", default=None, help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--fields", nargs="*", default=None, help="只导出指定字段；默认导出全部数值字段")
    parser.add_argument("--sample-instruments", type=int, default=0, help="仅导出前 N 只股票，用于调试")
    parser.add_argument("--include-aliases", action="store_true", help="额外导出 amount->money、turn_rate->turn/turnover 兼容别名")
    parser.add_argument("--clear-existing", action="store_true", help="导出前清空 calendars/instruments/features")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True, help="导出后检查 provider 结构")
    return parser.parse_args()


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    args = parse_args()
    dump_qlib_data(
        provider_uri=args.provider_uri,
        cache_dir=args.cache_dir,
        daily_dir=args.daily_dir,
        constituents_dir=args.constituents_dir,
        start=args.start,
        end=args.end,
        fields=args.fields,
        sample_instruments=args.sample_instruments,
        include_aliases=args.include_aliases,
        clear_existing=args.clear_existing,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
