"""Build and verify a local Qlib binary provider."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from data.config import CONFIG, PROJECT_ROOT, QLIB_FEATURE_FIELDS, DataConfig


def _date_string(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _write_bin(path: Path, values: np.ndarray, start_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.concatenate(
        (np.array([start_index], dtype="<f4"), values.astype("<f4", copy=False))
    )
    payload.tofile(path)


def _write_instruments(path: Path, spans: dict[str, list[tuple[str, str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{symbol}\t{start}\t{end}"
        for symbol in sorted(spans)
        for start, end in spans[symbol]
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    merged = [list(sorted(spans)[0])]
    for start, end in sorted(spans)[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _index_spans(
    weights: pd.DataFrame,
    calendar: list[str],
    allowed_symbols: set[str],
) -> dict[str, list[tuple[str, str]]]:
    if weights.empty:
        return {}
    calendar_dates = pd.DatetimeIndex(calendar)
    allowed_mask = np.fromiter(
        (str(symbol) in allowed_symbols for symbol in weights["symbol"]),
        dtype=bool,
        count=len(weights),
    )
    weights = weights.loc[allowed_mask].copy()
    weights["date"] = pd.to_datetime(weights["date"])
    snapshots = sorted(weights["date"].dropna().unique())
    ranges: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for number, snapshot in enumerate(snapshots):
        start = int(calendar_dates.searchsorted(snapshot, side="left"))
        if start >= len(calendar_dates):
            continue
        end = len(calendar_dates) - 1
        if number + 1 < len(snapshots):
            end = int(calendar_dates.searchsorted(snapshots[number + 1], side="left")) - 1
        if end < start:
            continue
        members = weights.loc[weights["date"].eq(snapshot), "symbol"].dropna().unique()
        for symbol in members:
            ranges[str(symbol)].append((start, end))
    return {
        symbol: [(calendar[start], calendar[end]) for start, end in _merge_spans(spans)]
        for symbol, spans in ranges.items()
    }


def _load_calendar(config: DataConfig, future: bool = False) -> list[str]:
    name = "trade_cal_future.parquet" if future else "trade_cal.parquet"
    path = config.raw_dir / "reference" / name
    if not path.is_file():
        raise FileNotFoundError(f"Trading calendar not found: {path}")
    frame = pd.read_parquet(path)
    dates = frame.loc[frame["is_open"].astype(str).eq("1"), "cal_date"].astype(str)
    end_date = config.future_calendar_end_date if future else config.end_date
    dates = dates[dates.between(config.start_date, end_date)]
    calendar = sorted(dates.map(lambda value: pd.Timestamp(value).strftime("%Y-%m-%d")).unique())
    if not calendar:
        raise ValueError("The configured period has no open trading days")
    return calendar


def _write_calendars(directory: Path, config: DataConfig) -> tuple[list[str], list[str]]:
    calendar = _load_calendar(config)
    future_calendar = _load_calendar(config, future=True)
    if future_calendar[: len(calendar)] != calendar:
        raise ValueError("Future Qlib calendar must contain the current calendar as an exact prefix")
    if len(future_calendar) <= len(calendar):
        raise ValueError("Future Qlib calendar must extend beyond the current calendar")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "day.txt").write_text("\n".join(calendar) + "\n", encoding="utf-8")
    (directory / "day_future.txt").write_text(
        "\n".join(future_calendar) + "\n", encoding="utf-8"
    )
    return calendar, future_calendar


def _write_features(
    stage: Path,
    stock_files: list[Path],
    calendar: list[str],
    config: DataConfig,
) -> dict[str, list[tuple[str, str]]]:
    calendar_index = pd.Index(calendar, name="date")
    calendar_set = set(calendar)
    spans: dict[str, list[tuple[str, str]]] = {}
    progress = tqdm(stock_files, desc="构建 Qlib", unit="只", dynamic_ncols=True)
    for path in progress:
        progress.set_postfix_str(path.stem, refresh=False)
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        missing = {"symbol", "date", "industry_code", *QLIB_FEATURE_FIELDS} - set(frame.columns)
        missing.discard("industry")
        if missing:
            raise ValueError(f"{path} is missing provider fields: {sorted(missing)}")
        dates = np.asarray(
            [_date_string(value) for value in frame["date"]],
            dtype=object,
        )
        date_mask = np.fromiter(
            (value in calendar_set for value in dates),
            dtype=bool,
            count=len(dates),
        )
        frame = frame.loc[date_mask].copy()
        frame["date"] = dates[date_mask]
        if frame.empty:
            continue
        if frame["date"].duplicated().any():
            raise ValueError(f"{path} contains duplicate trading dates")
        frame.sort_values("date", inplace=True)
        symbol_values = frame["symbol"].dropna().astype(str).unique()
        if len(symbol_values) != 1 or symbol_values[0] != path.stem:
            raise ValueError(f"{path} does not contain exactly its filename symbol")
        symbol = symbol_values[0]
        first_close = pd.to_numeric(frame["close"], errors="coerce").dropna()
        if first_close.empty or not np.isclose(
            first_close.iloc[0], 1.0, rtol=0.0, atol=config.price_normalization_tolerance
        ):
            raise ValueError(f"{symbol} first valid normalized close is not 1")

        spans[symbol] = [(frame["date"].iloc[0], frame["date"].iloc[-1])]
        aligned = frame.set_index("date").reindex(calendar_index)
        stock_start = calendar_index.get_loc(frame["date"].iloc[0])
        stock_end = calendar_index.get_loc(frame["date"].iloc[-1])
        for field in QLIB_FEATURE_FIELDS:
            source = "industry_code" if field == "industry" else field
            values = pd.to_numeric(aligned[source], errors="coerce").to_numpy(dtype=np.float32)
            valid = np.flatnonzero(~np.isnan(values))
            start = int(valid[0]) if valid.size else int(stock_start)
            end = int(valid[-1]) if valid.size else int(stock_end)
            _write_bin(
                stage / "features" / symbol.lower() / f"{field}.day.bin",
                values[start : end + 1],
                start,
            )
    return spans


def _publish(stage: Path, target: Path, replace_existing: bool, data_root: Path) -> None:
    target = target.expanduser().resolve()
    if target == Path(target.anchor):
        raise ValueError(f"Refusing to publish a provider at filesystem root: {target}")
    protected = (PROJECT_ROOT.resolve(), data_root.expanduser().resolve())
    if any(
        target == path or target in path.parents or path in target.parents
        for path in protected
    ):
        raise ValueError(f"Refusing to replace a protected project/data directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_name(f".{target.name}.backup-{uuid4().hex}")
    if target.exists():
        if not replace_existing:
            raise FileExistsError(f"Qlib provider already exists: {target}")
        if target.is_symlink() or not target.is_dir():
            raise ValueError(f"Refusing to replace a non-directory provider path: {target}")
        entries = list(target.iterdir())
        is_provider = all(
            required.exists()
            for required in (
                target / "calendars" / "day.txt",
                target / "instruments" / "all.txt",
                target / "features",
            )
        )
        if entries and not is_provider:
            raise ValueError(f"Refusing to replace an unrecognized non-empty directory: {target}")
        target.replace(backup)
    try:
        stage.replace(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.replace(target)
        raise
    if backup.exists():
        try:
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
        except OSError as exc:
            tqdm.write(f"警告：新 provider 已生效，但旧备份无法删除：{exc}")


def build_provider(config: DataConfig = CONFIG) -> Path:
    config.validate()
    manifest_path = config.standard_dir / "_SUCCESS.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Standardized data success manifest is missing; normalize first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("start_date") != config.start_date or manifest.get("end_date") != config.end_date:
        raise ValueError("Standardized data success manifest does not match the configured date range")
    expected = set(manifest.get("stocks", []))
    nonempty = set(manifest.get("nonempty_stocks", []))
    available = {path.stem for path in (config.standard_dir / "stocks").glob("*.parquet")}
    if available != expected:
        raise ValueError(
            "Standardized stock files do not match the success manifest: "
            f"missing={sorted(expected - available)}, extra={sorted(available - expected)}"
        )
    if len(nonempty) != manifest.get("nonempty_stock_count") or not nonempty.issubset(expected):
        raise ValueError("Standardized data success manifest has inconsistent stock counts")
    stock_files = [config.standard_dir / "stocks" / f"{symbol}.parquet" for symbol in sorted(nonempty)]
    if not stock_files:
        raise ValueError("Standardized data success manifest contains no non-empty stocks")
    target = config.provider_uri.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(f".{target.name}.build-{uuid4().hex}")
    stage.mkdir()
    try:
        calendar, _ = _write_calendars(stage / "calendars", config)
        spans = _write_features(stage, stock_files, calendar, config)
        if set(spans) != nonempty:
            raise ValueError("Built Qlib instruments do not match the standardized success manifest")
        _write_instruments(stage / "instruments" / "all.txt", spans)

        for index in config.indices:
            weight_path = config.standard_dir / "index_weights" / f"{index.market}.parquet"
            if not weight_path.is_file():
                raise FileNotFoundError(f"Standardized index weights not found: {weight_path}")
            index_members = _index_spans(pd.read_parquet(weight_path), calendar, set(spans))
            if not index_members:
                raise ValueError(f"No Qlib instrument spans could be built for {index.market}")
            _write_instruments(
                stage / "instruments" / f"{index.market}.txt", index_members
            )

        metadata = {
            "source": "Tushare",
            "start_date": config.start_date,
            "end_date": config.end_date,
            "future_calendar_end_date": config.future_calendar_end_date,
            "instrument_count": len(spans),
            "industry_taxonomy": "SW2021_L1",
            "features": list(QLIB_FEATURE_FIELDS),
        }
        (stage / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        if config.run_verify:
            verify_provider(stage, config)
        _publish(stage, target, config.replace_existing_provider, config.data_root)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return target


def refresh_provider_symbols(symbols: set[str], config: DataConfig = CONFIG) -> Path:
    """Atomically refresh selected feature directories in an existing provider."""
    if not symbols:
        return config.provider_uri.expanduser().resolve()

    target = config.provider_uri.expanduser().resolve()
    if not all(
        path.exists()
        for path in (
            target / "calendars" / "day.txt",
            target / "instruments" / "all.txt",
            target / "features",
        )
    ):
        raise FileNotFoundError(f"Qlib provider is incomplete: {target}")

    manifest_path = config.standard_dir / "_SUCCESS.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nonempty = set(map(str, manifest.get("nonempty_stocks", [])))
    missing = symbols - nonempty
    if missing:
        raise ValueError(f"Requested symbols are absent from standardized data: {sorted(missing)}")

    calendar = _load_calendar(config)
    stock_files = [config.standard_dir / "stocks" / f"{symbol}.parquet" for symbol in sorted(symbols)]
    with tempfile.TemporaryDirectory(prefix=".provider-refresh-", dir=config.data_root) as temporary:
        stage = Path(temporary)
        refreshed_spans = _write_features(stage, stock_files, calendar, config)
        if set(refreshed_spans) != symbols:
            raise ValueError("Refreshed Qlib instruments do not match requested symbols")

        existing = pd.read_csv(
            target / "instruments" / "all.txt",
            sep="\t",
            header=None,
            names=["symbol", "start", "end"],
            dtype=str,
        )
        all_spans: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for row in existing.itertuples(index=False):
            if str(row.symbol) not in symbols:
                all_spans[str(row.symbol)].append((str(row.start), str(row.end)))
        all_spans.update(refreshed_spans)

        _write_instruments(stage / "instruments" / "all.txt", all_spans)
        for index in config.indices:
            weights = pd.read_parquet(
                config.standard_dir / "index_weights" / f"{index.market}.parquet"
            )
            members = _index_spans(weights, calendar, set(all_spans))
            if not members:
                raise ValueError(f"No Qlib instrument spans could be built for {index.market}")
            _write_instruments(stage / "instruments" / f"{index.market}.txt", members)

        metadata = {
            "source": "Tushare",
            "start_date": config.start_date,
            "end_date": config.end_date,
            "future_calendar_end_date": config.future_calendar_end_date,
            "instrument_count": len(all_spans),
            "industry_taxonomy": "SW2021_L1",
            "features": list(QLIB_FEATURE_FIELDS),
        }
        (stage / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8"
        )

        replacements = [
            (source, target / source.relative_to(stage))
            for symbol in sorted(symbols)
            for source in (stage / "features" / symbol.lower()).glob("*.day.bin")
        ]
        replacements.extend(
            (source, target / source.relative_to(stage))
            for source in (stage / "instruments").glob("*.txt")
        )
        replacements.append((stage / "metadata.json", target / "metadata.json"))
        for source, destination in replacements:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
            shutil.copyfile(source, temporary_path)
            temporary_path.replace(destination)

    verify_provider(target, config)
    return target


def refresh_provider_calendars(config: DataConfig = CONFIG) -> Path:
    """Atomically refresh day.txt and day_future.txt in an existing provider."""
    target = config.provider_uri.expanduser().resolve()
    if not (target / "calendars" / "day.txt").is_file():
        raise FileNotFoundError(f"Qlib provider calendar is missing: {target}")
    with tempfile.TemporaryDirectory(prefix=".calendar-refresh-", dir=config.data_root) as temporary:
        stage = Path(temporary)
        _write_calendars(stage, config)
        for source in stage.glob("*.txt"):
            destination = target / "calendars" / source.name
            temporary_path = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
            shutil.copyfile(source, temporary_path)
            temporary_path.replace(destination)
        metadata_path = target / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["future_calendar_end_date"] = config.future_calendar_end_date
            temporary_path = metadata_path.with_name(f".{metadata_path.name}.{uuid4().hex}.tmp")
            temporary_path.write_text(
                json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8"
            )
            temporary_path.replace(metadata_path)
    verify_provider(target, config)
    return target


def _read_bin(path: Path) -> tuple[int, np.ndarray]:
    payload = np.fromfile(path, dtype="<f4")
    if payload.size < 2 or not np.isfinite(payload[0]) or payload[0] != int(payload[0]):
        raise ValueError(f"Invalid Qlib feature payload: {path}")
    return int(payload[0]), payload[1:]


def verify_provider(provider_uri: str | Path, config: DataConfig = CONFIG) -> None:
    provider = Path(provider_uri).expanduser().resolve()
    calendar_path = provider / "calendars" / "day.txt"
    if not calendar_path.is_file():
        raise FileNotFoundError(f"Missing Qlib calendar: {calendar_path}")
    calendar = [line for line in calendar_path.read_text(encoding="utf-8").splitlines() if line]
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError("Qlib day calendar must be non-empty, sorted, and unique")
    future_path = provider / "calendars" / "day_future.txt"
    if not future_path.is_file():
        raise FileNotFoundError(f"Missing Qlib future calendar: {future_path}")
    future_calendar = [line for line in future_path.read_text(encoding="utf-8").splitlines() if line]
    if future_calendar != sorted(set(future_calendar)):
        raise ValueError("Qlib future calendar must be sorted and unique")
    if future_calendar[: len(calendar)] != calendar or len(future_calendar) <= len(calendar):
        raise ValueError("Qlib future calendar must extend the complete current calendar")

    markets = ("all", *(index.market for index in config.indices))
    for market in markets:
        path = provider / "instruments" / f"{market}.txt"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty Qlib market instruments: {path}")
        instruments = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["symbol", "start", "end"],
            dtype={"symbol": str},
        )
        if instruments[["symbol", "start", "end"]].isna().any(axis=None):
            raise ValueError(f"Invalid instrument rows in {path}")
        if instruments["start"].gt(instruments["end"]).any():
            raise ValueError(f"Instrument start is after end in {path}")
        if not instruments["start"].isin(calendar).all() or not instruments["end"].isin(calendar).all():
            raise ValueError(f"Instrument dates are outside the Qlib calendar in {path}")

    all_instruments = pd.read_csv(
        provider / "instruments" / "all.txt",
        sep="\t",
        header=None,
        names=["symbol", "start", "end"],
        dtype={"symbol": str},
    )
    manifest_path = config.standard_dir / "_SUCCESS.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = set(map(str, manifest.get("nonempty_stocks", [])))
        actual = set(all_instruments["symbol"].astype(str))
        if actual != expected:
            raise ValueError(
                "Qlib instruments do not match the standardized success manifest: "
                f"missing={sorted(expected - actual)[:10]}, extra={sorted(actual - expected)[:10]}"
            )
    for symbol in all_instruments["symbol"].unique():
        feature_dir = provider / "features" / symbol.lower()
        for field in QLIB_FEATURE_FIELDS:
            path = feature_dir / f"{field}.day.bin"
            if not path.is_file():
                raise FileNotFoundError(f"Missing required Qlib feature: {path}")
            start, values = _read_bin(path)
            if start < 0 or start + len(values) > len(calendar):
                raise ValueError(f"Qlib feature exceeds calendar bounds: {path}")
        close_path = feature_dir / "close.day.bin"
        _, close_values = _read_bin(close_path)
        first_close = close_values[np.flatnonzero(~np.isnan(close_values))[0]]
        if not np.isclose(
            first_close, 1.0, rtol=0.0, atol=config.price_normalization_tolerance
        ):
            raise ValueError(f"{symbol} first valid Qlib close is not 1")


def main() -> None:
    build_provider()


if __name__ == "__main__":
    main()
