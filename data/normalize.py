"""Normalize raw Tushare data into per-instrument Parquet datasets."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from data.config import CONFIG, STANDARD_COLUMNS, DataConfig
from data.download import atomic_parquet


def tushare_to_qlib_symbol(ts_code: str) -> str:
    try:
        code, exchange = str(ts_code).upper().split(".")
    except ValueError as exc:
        raise ValueError(f"Invalid Tushare stock code: {ts_code}") from exc
    prefixes = {"SH": "SH", "SZ": "SZ", "BJ": "BJ"}
    if exchange not in prefixes:
        raise ValueError(f"Unsupported Tushare stock exchange: {ts_code}")
    return prefixes[exchange] + code


def _read_chunks(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw Parquet files found in {directory}")
    frames = [pd.read_parquet(path) for path in files]
    return pd.concat(frames, ignore_index=True, sort=False)


def _deduplicate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.drop_duplicates(keys)
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        compare_columns = [column for column in frame.columns if column not in keys]
        conflicts = frame.loc[duplicates].groupby(keys, dropna=False)[compare_columns].nunique(
            dropna=False
        )
        if conflicts.gt(1).any(axis=None):
            raise ValueError(f"Conflicting duplicate raw rows for keys {keys}")
    return frame.drop_duplicates(keys, keep="last")


def _numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _industry_history(config: DataConfig, industry_codes: set[str]) -> pd.DataFrame:
    files = [
        config.raw_dir / "index_member_all" / code / f"{name}.parquet"
        for code in sorted(industry_codes)
        for name in ("historical", "current")
    ]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing industry member caches: {missing[:5]}")
    history = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True, sort=False)
    if history.empty:
        return history
    history = history.dropna(subset=["l1_code", "in_date"]).copy()
    history["in_date"] = pd.to_datetime(history["in_date"], format="%Y%m%d", errors="coerce")
    history["out_date"] = pd.to_datetime(history["out_date"], format="%Y%m%d", errors="coerce")
    history = history.dropna(subset=["in_date"])
    return _deduplicate(history, ["ts_code", "l1_code", "in_date", "out_date"])


def _add_industry(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    valid_industries: set[str],
) -> pd.DataFrame:
    frame["industry"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame["industry_code"] = np.nan
    if history.empty:
        return frame

    history = history[history["l1_code"].astype(str).isin(valid_industries)].copy()
    history.sort_values(["in_date", "out_date"], inplace=True, na_position="last")
    dates = frame["date"]
    for row in history.itertuples(index=False):
        end = row.out_date if pd.notna(row.out_date) else pd.Timestamp.max.normalize()
        mask = dates.ge(row.in_date) & dates.le(end)
        code = str(row.l1_code).split(".", maxsplit=1)[0]
        frame.loc[mask, "industry"] = str(row.l1_name)
        frame.loc[mask, "industry_code"] = pd.to_numeric(code, errors="coerce")
    return frame


def normalize_stock(
    config: DataConfig,
    ts_code: str,
    valid_industries: set[str],
    industry_history: pd.DataFrame,
) -> pd.DataFrame:
    root = config.raw_dir / "stocks"
    daily = _deduplicate(_read_chunks(root / "daily" / ts_code), ["ts_code", "trade_date"])
    factors = _deduplicate(
        _read_chunks(root / "adj_factor" / ts_code), ["ts_code", "trade_date"]
    )
    basics = _deduplicate(
        _read_chunks(root / "daily_basic" / ts_code), ["ts_code", "trade_date"]
    )
    if daily.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    frame = daily.merge(factors, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    frame = frame.merge(
        basics.drop(columns=["close"], errors="ignore"),
        on=["ts_code", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    frame["date"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d", errors="raise")
    frame = frame[
        frame["date"].between(pd.Timestamp(config.start_date), pd.Timestamp(config.end_date))
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    frame.sort_values("date", inplace=True)
    frame.reset_index(drop=True, inplace=True)

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "pct_chg",
        "vol",
        "amount",
        "adj_factor",
        "turnover_rate",
        "volume_ratio",
        "pe_ttm",
        "ps_ttm",
        "dv_ttm",
        "total_mv",
        "circ_mv",
        "limit_status",
    ]
    _numeric(frame, numeric_columns)
    if frame["adj_factor"].isna().any() or frame["adj_factor"].le(0).any():
        missing_dates = frame.loc[
            frame["adj_factor"].isna() | frame["adj_factor"].le(0), "trade_date"
        ].tolist()
        raise ValueError(f"{ts_code} has invalid adjustment factors on {missing_dates[:5]}")

    adjusted_close = frame["close"] * frame["adj_factor"]
    valid_close = adjusted_close.dropna()
    if valid_close.empty or valid_close.iloc[0] <= 0:
        raise ValueError(f"{ts_code} has no positive adjusted close")
    first_adjusted_close = float(valid_close.iloc[0])
    raw_close = frame["close"].copy()
    frame["factor"] = frame["adj_factor"] / first_adjusted_close
    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column] * frame["factor"]

    raw_volume = frame["vol"]
    raw_vwap = frame["amount"] * 10.0 / raw_volume.where(raw_volume.gt(0))
    frame["volume"] = raw_volume / frame["factor"]
    frame["vwap"] = raw_vwap * frame["factor"]
    frame["change"] = raw_close.pct_change(fill_method=None)
    frame["turnover"] = frame["turnover_rate"]

    frame["symbol"] = tushare_to_qlib_symbol(ts_code)
    frame = _add_industry(frame, industry_history, valid_industries)
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame.loc[:, STANDARD_COLUMNS]


def _normalize_index_weights(config: DataConfig) -> None:
    output_dir = config.standard_dir / "index_weights"
    for index in config.indices:
        raw_dir = config.raw_dir / "index_weight" / index.market
        weights = _read_chunks(raw_dir)
        if weights.empty:
            tqdm.write(f"警告：{index.market} 没有可用的指数权重")
            weights = pd.DataFrame(columns=["market", "symbol", "date", "weight"])
        else:
            weights = _deduplicate(weights, ["index_code", "con_code", "trade_date"])
            weights["market"] = index.market
            weights["symbol"] = weights["con_code"].map(tushare_to_qlib_symbol)
            weights["date"] = pd.to_datetime(
                weights["trade_date"], format="%Y%m%d", errors="raise"
            )
            weights = weights[
                weights["date"].between(
                    pd.Timestamp(config.start_date), pd.Timestamp(config.end_date)
                )
            ].copy()
            weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce")
            weights = weights.loc[:, ["market", "symbol", "date", "weight"]]
            weights.sort_values(["date", "symbol"], inplace=True)
        atomic_parquet(weights, output_dir / f"{index.market}.parquet")


def _standard_is_current(
    path: Path,
    raw_root: Path,
    ts_code: str,
    config: DataConfig,
    industry_updated_ns: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return False
    if not set(STANDARD_COLUMNS).issubset(frame.columns):
        return False
    if not frame.empty:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if dates.isna().any():
            return False
        if dates.min() < pd.Timestamp(config.start_date) or dates.max() > pd.Timestamp(config.end_date):
            return False
    daily_files = sorted((raw_root / "daily" / ts_code).glob("*.parquet"))
    if not daily_files:
        return False
    raw_trade_dates = pd.concat(
        [
            pd.read_parquet(daily_file, columns=["trade_date"])["trade_date"]
            for daily_file in daily_files
        ],
        ignore_index=True,
    ).dropna()
    raw_trade_dates = raw_trade_dates.astype(str)
    raw_dates = set(
        pd.to_datetime(
            raw_trade_dates[raw_trade_dates.between(config.start_date, config.end_date)],
            format="%Y%m%d",
            errors="raise",
        )
    )
    normalized_dates = set(pd.to_datetime(frame["date"], errors="coerce"))
    if raw_dates != normalized_dates:
        return False
    raw_files = [
        raw_path
        for endpoint in ("daily", "adj_factor", "daily_basic")
        for raw_path in (raw_root / endpoint / ts_code).glob("*.parquet")
    ]
    raw_files.append(config.raw_dir / "reference" / "sw2021_l1.parquet")
    raw_files = [raw for raw in raw_files if raw.is_file()]
    latest_raw = max(
        max(raw.stat().st_mtime_ns for raw in raw_files),
        industry_updated_ns,
    )
    return bool(raw_files) and latest_raw <= path.stat().st_mtime_ns


def normalize_all(config: DataConfig = CONFIG) -> None:
    config.validate()
    marker = config.standard_dir / "_SUCCESS.json"
    marker.unlink(missing_ok=True)
    raw_marker = config.raw_dir / "_SUCCESS.json"
    if not raw_marker.is_file():
        raise FileNotFoundError("Raw download success manifest is missing; complete the download first")
    raw_manifest = json.loads(raw_marker.read_text(encoding="utf-8"))
    if raw_manifest.get("start_date") != config.start_date or raw_manifest.get("end_date") != config.end_date:
        raise ValueError("Raw download success manifest does not match the configured date range")
    universe_name = "stock_universe_sample.parquet" if config.sample_ts_codes else "stock_universe.parquet"
    universe_path = config.raw_dir / "reference" / universe_name
    if not universe_path.is_file():
        raise FileNotFoundError("No stock universe found; run the download stage first")
    stocks = pd.read_parquet(universe_path).drop_duplicates("ts_code", keep="first")
    stocks = stocks[
        stocks["exchange"].isin(config.stock_exchanges)
        & stocks["market"].isin(config.stock_markets)
        & stocks["curr_type"].eq("CNY")
        & stocks["list_date"].fillna("99999999").astype(str).le(config.end_date)
        & (
            stocks["delist_date"].isna()
            | stocks["delist_date"].astype(str).ge(config.start_date)
        )
    ].sort_values("ts_code")
    manifest_stocks = set(map(str, raw_manifest.get("stocks", [])))
    if set(stocks["ts_code"].astype(str)) != manifest_stocks:
        raise ValueError("Raw stock universe does not match the download success manifest")

    classes = pd.read_parquet(config.raw_dir / "reference" / "sw2021_l1.parquet")
    valid_industries = set(classes["index_code"].dropna().astype(str))
    if not valid_industries:
        raise ValueError("The SW2021 L1 classification is empty")
    if set(map(str, raw_manifest.get("industry_codes", []))) != valid_industries:
        raise ValueError("Industry member caches do not match the download success manifest")
    industry_history = _industry_history(config, valid_industries)
    industry_by_stock = {
        str(ts_code): group.copy()
        for ts_code, group in industry_history.groupby("ts_code", sort=False)
    }
    industry_files = list((config.raw_dir / "index_member_all").glob("*/*.parquet"))
    industry_updated_ns = max(path.stat().st_mtime_ns for path in industry_files)

    output_dir = config.standard_dir / "stocks"
    total = len(stocks)
    nonempty = 0
    nonempty_symbols = []
    expected_symbols = [tushare_to_qlib_symbol(code) for code in stocks["ts_code"].astype(str)]
    progress = tqdm(
        list(stocks.itertuples(index=False)),
        desc="标准化股票",
        unit="只",
        dynamic_ncols=True,
    )
    for stock in progress:
        progress.set_postfix_str(str(stock.ts_code), refresh=False)
        output = output_dir / f"{tushare_to_qlib_symbol(stock.ts_code)}.parquet"
        if config.resume and _standard_is_current(
            output,
            config.raw_dir / "stocks",
            str(stock.ts_code),
            config,
            industry_updated_ns,
        ):
            if not pd.read_parquet(output).empty:
                nonempty += 1
                nonempty_symbols.append(output.stem)
            continue
        normalized = normalize_stock(
            config,
            str(stock.ts_code),
            valid_industries,
            industry_by_stock.get(str(stock.ts_code), pd.DataFrame()),
        )
        atomic_parquet(normalized, output)
        nonempty += int(not normalized.empty)
        if not normalized.empty:
            nonempty_symbols.append(output.stem)

    _normalize_index_weights(config)
    expected_set = set(expected_symbols)
    for stale in output_dir.glob("*.parquet"):
        if stale.stem not in expected_set:
            stale.unlink()
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_name(f".{marker.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "start_date": config.start_date,
                "end_date": config.end_date,
                "stock_count": total,
                "nonempty_stock_count": nonempty,
                "stocks": expected_symbols,
                "nonempty_stocks": nonempty_symbols,
                "industry_taxonomy": "SW2021_L1",
                "factor_semantics": "qlib_adjusted_price_over_raw_price",
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(marker)


def main() -> None:
    normalize_all()


if __name__ == "__main__":
    main()
