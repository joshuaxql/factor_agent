"""Resumable Tushare downloads stored as raw Parquet chunks."""

from __future__ import annotations

import calendar
import json
import random
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import pandas as pd
from tqdm.auto import tqdm

from data.config import (
    CONFIG,
    ENDPOINT_FIELDS,
    ENDPOINT_ROW_LIMITS,
    STOCK_FIELDS,
    SW_CLASS_FIELDS,
    TRADE_CAL_FIELDS,
    DataConfig,
)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def _format_date(value: date) -> str:
    return value.strftime("%Y%m%d")


def _add_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, day=28)


def date_windows(start: str, end: str, years: int) -> Iterable[tuple[str, str]]:
    current = _parse_date(start)
    final = _parse_date(end)
    while current <= final:
        window_end = min(_add_years(current, years) - timedelta(days=1), final)
        yield _format_date(current), _format_date(window_end)
        current = window_end + timedelta(days=1)


def month_windows(start: str, end: str) -> Iterable[tuple[str, str, str]]:
    current = _parse_date(start).replace(day=1)
    final = _parse_date(end)
    while current <= final:
        month_end = current.replace(day=calendar.monthrange(current.year, current.month)[1])
        window_start = max(current, _parse_date(start))
        window_end = min(month_end, final)
        yield current.strftime("%Y%m"), _format_date(window_start), _format_date(window_end)
        current = (month_end + timedelta(days=1)).replace(day=1)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_valid_parquet(path: Path, required_columns: Iterable[str]) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        tqdm.write(f"缓存损坏，将重新下载：{path}（{exc}）")
        return None
    missing = set(required_columns) - set(frame.columns)
    if missing:
        tqdm.write(f"缓存字段不完整，将重新下载：{path}（缺少 {sorted(missing)}）")
        return None
    return frame


class TushareClient:
    """Rate-limited Tushare client with exponential retry."""

    def __init__(self, pro: Any, config: DataConfig = CONFIG):
        self.pro = pro
        self.config = config
        self._last_request = 0.0

    def request(
        self,
        endpoint: str,
        fields: str,
        row_limit: int | None = None,
        **parameters: Any,
    ) -> pd.DataFrame:
        expected = fields.split(",")
        for attempt in range(self.config.max_retries):
            wait = self.config.request_interval_seconds - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                self._last_request = time.monotonic()
                method = getattr(self.pro, endpoint)
                result = method(fields=fields, **parameters)
                frame = pd.DataFrame() if result is None else pd.DataFrame(result)
                if frame.empty:
                    return pd.DataFrame(columns=expected)
                missing = set(expected) - set(frame.columns)
                if missing:
                    raise ValueError(f"{endpoint} response is missing columns: {sorted(missing)}")
                if row_limit is not None and len(frame) >= row_limit:
                    raise RuntimeError(
                        f"{endpoint} returned {len(frame)} rows, at its documented limit; "
                        "reduce the configured download window to avoid truncation"
                    )
                return frame.loc[:, expected]
            except Exception as exc:  # noqa: BLE001
                if attempt + 1 >= self.config.max_retries:
                    raise RuntimeError(
                        f"Tushare {endpoint} failed after {self.config.max_retries} attempts: {parameters}"
                    ) from exc
                delay = self.config.retry_delay_seconds * (2**attempt) + random.uniform(0.0, 0.25)
                tqdm.write(
                    f"Tushare {endpoint} 第 {attempt + 1}/{self.config.max_retries} 次请求失败："
                    f"{exc}；{delay:.2f} 秒后重试"
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def query(self, endpoint: str, **parameters: Any) -> pd.DataFrame:
        return self.request(
            endpoint,
            ENDPOINT_FIELDS[endpoint],
            ENDPOINT_ROW_LIMITS.get(endpoint),
            **parameters,
        )

    def query_pages(self, endpoint: str, page_size: int, **parameters: Any) -> pd.DataFrame:
        frames = []
        offset = 0
        while True:
            page = self.request(
                endpoint,
                ENDPOINT_FIELDS[endpoint],
                limit=page_size,
                offset=offset,
                **parameters,
            )
            frames.append(page)
            if len(page) < page_size:
                break
            offset += page_size
        return pd.concat(frames, ignore_index=True, sort=False)


class Downloader:
    def __init__(self, config: DataConfig = CONFIG, pro: Any | None = None):
        self.config = config
        if pro is None:
            if not config.tushare_token:
                raise RuntimeError("TUSHARE_TOKEN is required when run_download is enabled")
            try:
                import tushare as ts
            except ImportError as exc:
                raise RuntimeError("Install the tushare dependency before downloading data") from exc
            pro = ts.pro_api(
                config.tushare_token,
                timeout=config.request_timeout_seconds,
            )
        self.client = TushareClient(pro, config)

    def _download_trade_calendars(self, reference_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        calendar_specs = (
            ("trade_cal", self.config.end_date),
            ("trade_cal_future", self.config.future_calendar_end_date),
        )
        calendars = []
        cal_columns = TRADE_CAL_FIELDS.split(",")
        for name, end_date in calendar_specs:
            path = reference_dir / f"{name}.parquet"
            metadata_path = reference_dir / f"{name}.json"
            cached = read_valid_parquet(path, cal_columns) if self.config.resume else None
            if cached is not None:
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (FileNotFoundError, json.JSONDecodeError):
                    cached = None
                else:
                    if metadata != {"start_date": self.config.start_date, "end_date": end_date}:
                        cached = None
            if cached is None:
                cached = self.client.request(
                    "trade_cal",
                    TRADE_CAL_FIELDS,
                    exchange="SSE",
                    start_date=self.config.start_date,
                    end_date=end_date,
                    is_open="1",
                )
                if cached.empty:
                    raise RuntimeError(f"trade_cal returned no open trading days through {end_date}")
                atomic_parquet(cached, path)
                metadata_path.write_text(
                    json.dumps(
                        {"start_date": self.config.start_date, "end_date": end_date},
                        ensure_ascii=True,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            calendars.append(cached)

        current, future = calendars
        current_dates = set(current["cal_date"].dropna().astype(str))
        future_dates = set(future["cal_date"].dropna().astype(str))
        if not current_dates.issubset(future_dates):
            raise RuntimeError("Future trade calendar does not contain the complete current calendar")
        if not any(date > self.config.end_date for date in future_dates):
            raise RuntimeError("Future trade calendar contains no trading day after end_date")
        return current, future

    def _cached_query(
        self,
        path: Path,
        endpoint: str,
        required_trade_dates: Iterable[str] | None = None,
        **parameters: Any,
    ) -> pd.DataFrame:
        expected = ENDPOINT_FIELDS[endpoint].split(",")
        required_dates = (
            None
            if required_trade_dates is None
            else {str(value) for value in required_trade_dates if pd.notna(value)}
        )

        def missing_dates(frame: pd.DataFrame) -> list[str]:
            if required_dates is None:
                return []
            available = set(frame["trade_date"].dropna().astype(str))
            return sorted(required_dates - available)

        if self.config.resume:
            cached = read_valid_parquet(path, expected)
            if cached is not None:
                missing = missing_dates(cached)
                if not missing:
                    return cached
                tqdm.write(f"缓存交易日不完整，将重新下载：{path}（缺少 {missing[:5]}）")
        frame = self.client.query(endpoint, **parameters)
        missing = missing_dates(frame)
        if missing:
            raise RuntimeError(
                f"Tushare {endpoint} response for {parameters.get('ts_code', path.parent.name)} "
                f"is missing trade dates: {missing[:5]}"
            )
        atomic_parquet(frame, path)
        return frame

    def _download_reference(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        reference_dir = self.config.raw_dir / "reference"
        stock_frames = []
        stock_columns = STOCK_FIELDS.split(",")
        for status in self.config.stock_statuses:
            for exchange in self.config.stock_exchanges:
                path = reference_dir / f"stock_basic_{status}_{exchange}.parquet"
                cached = read_valid_parquet(path, stock_columns) if self.config.resume else None
                if cached is None:
                    cached = self.client.request(
                        "stock_basic",
                        STOCK_FIELDS,
                        row_limit=6000,
                        exchange=exchange,
                        list_status=status,
                    )
                    atomic_parquet(cached, path)
                stock_frames.append(cached)

        stocks = pd.concat(stock_frames, ignore_index=True).drop_duplicates("ts_code", keep="first")
        stocks = stocks[
            stocks["exchange"].isin(self.config.stock_exchanges)
            & stocks["market"].isin(self.config.stock_markets)
            & stocks["curr_type"].eq("CNY")
            & stocks["list_date"].fillna("99999999").astype(str).le(self.config.end_date)
            & (
                stocks["delist_date"].isna()
                | stocks["delist_date"].astype(str).ge(self.config.start_date)
            )
        ]
        if self.config.sample_ts_codes:
            requested = set(self.config.sample_ts_codes)
            stocks = stocks[stocks["ts_code"].isin(requested)].copy()
            missing = requested - set(stocks["ts_code"])
            if missing:
                raise ValueError(f"Sample stocks are absent from stock_basic: {sorted(missing)}")
        stocks.sort_values("ts_code", inplace=True)
        universe_name = "stock_universe.parquet"
        if self.config.sample_ts_codes:
            universe_name = "stock_universe_sample.parquet"
        atomic_parquet(stocks, reference_dir / universe_name)

        cached_cal, _ = self._download_trade_calendars(reference_dir)

        class_path = reference_dir / "sw2021_l1.parquet"
        class_columns = SW_CLASS_FIELDS.split(",")
        cached_class = read_valid_parquet(class_path, class_columns) if self.config.resume else None
        if cached_class is None:
            cached_class = self.client.request(
                "index_classify", SW_CLASS_FIELDS, level="L1", src="SW2021"
            )
            if cached_class.empty:
                raise RuntimeError("index_classify returned no SW2021 level-1 industries")
            atomic_parquet(cached_class, class_path)

        return stocks, cached_class

    def _download_industry_members(self, classes: pd.DataFrame) -> list[str]:
        codes = sorted(classes["index_code"].dropna().astype(str).unique())
        tasks = [
            (code, is_new, name)
            for code in codes
            for is_new, name in (("N", "historical"), ("Y", "current"))
        ]
        progress = tqdm(tasks, desc="下载申万行业", unit="类", dynamic_ncols=True)
        expected = ENDPOINT_FIELDS["index_member_all"].split(",")
        for code, is_new, name in progress:
            progress.set_postfix_str(f"{code} {name}", refresh=False)
            path = self.config.raw_dir / "index_member_all" / code / f"{name}.parquet"
            cached = read_valid_parquet(path, expected) if self.config.resume else None
            if cached is None:
                cached = self.client.query_pages(
                    "index_member_all",
                    ENDPOINT_ROW_LIMITS["index_member_all"],
                    l1_code=code,
                    is_new=is_new,
                )
                atomic_parquet(cached, path)
        return codes

    def _stock_range(self, stock: Any) -> tuple[str, str] | None:
        start = max(self.config.start_date, str(stock.list_date))
        delist_date = stock.delist_date
        end = self.config.end_date
        if pd.notna(delist_date) and str(delist_date):
            end = min(end, str(delist_date))
        return (start, end) if start <= end else None

    def _download_stock(self, stock: Any, progress: Any | None = None) -> None:
        stock_range = self._stock_range(stock)
        if stock_range is None:
            return
        start, end = stock_range
        symbol = str(stock.ts_code)
        for window_start, window_end in date_windows(start, end, self.config.download_window_years):
            window_name = f"{window_start}_{window_end}.parquet"

            basic_path = (
                self.config.raw_dir / "stocks" / "daily_basic" / symbol / window_name
            )
            cached_basic = read_valid_parquet(
                basic_path,
                ENDPOINT_FIELDS["daily_basic"].split(","),
            )

            daily_path = self.config.raw_dir / "stocks" / "daily" / symbol / window_name
            if progress is not None:
                progress.set_postfix_str(
                    f"{symbol} daily {window_start}-{window_end}", refresh=True
                )
            daily = self._cached_query(
                daily_path,
                "daily",
                required_trade_dates=None if cached_basic is None else cached_basic["trade_date"],
                ts_code=symbol,
                start_date=window_start,
                end_date=window_end,
            )

            factor_path = (
                self.config.raw_dir / "stocks" / "adj_factor" / symbol / window_name
            )
            if progress is not None:
                progress.set_postfix_str(
                    f"{symbol} adj_factor {window_start}-{window_end}", refresh=True
                )
            self._cached_query(
                factor_path,
                "adj_factor",
                required_trade_dates=daily["trade_date"],
                ts_code=symbol,
                start_date=window_start,
                end_date=window_end,
            )

            if progress is not None:
                progress.set_postfix_str(
                    f"{symbol} daily_basic {window_start}-{window_end}", refresh=True
                )
            self._cached_query(
                basic_path,
                "daily_basic",
                required_trade_dates=daily["trade_date"],
                ts_code=symbol,
                start_date=window_start,
                end_date=window_end,
            )

    def _download_index_weights(self) -> None:
        for index in self.config.indices:
            target_dir = self.config.raw_dir / "index_weight" / index.market
            metadata_path = target_dir / "_SOURCE.json"
            cached_source = None
            try:
                cached_source = json.loads(metadata_path.read_text(encoding="utf-8")).get(
                    "index_code"
                )
            except (FileNotFoundError, json.JSONDecodeError, AttributeError):
                pass

            if cached_source is None:
                observed_codes = set()
                for path in target_dir.glob("*.parquet"):
                    cached = read_valid_parquet(
                        path, ENDPOINT_FIELDS["index_weight"].split(",")
                    )
                    if cached is not None:
                        observed_codes.update(cached["index_code"].dropna().astype(str))
                cache_matches_source = observed_codes == {index.tushare_code}
            else:
                cache_matches_source = cached_source == index.tushare_code

            windows = list(month_windows(self.config.start_date, self.config.end_date))
            progress = tqdm(
                windows,
                desc=f"指数权重 {index.market}",
                unit="月",
                dynamic_ncols=True,
            )
            for month, start, end in progress:
                progress.set_postfix_str(month, refresh=False)
                path = target_dir / f"{month}.parquet"
                expected = ENDPOINT_FIELDS["index_weight"].split(",")
                cached = (
                    read_valid_parquet(path, expected)
                    if self.config.resume and cache_matches_source
                    else None
                )
                if cached is None:
                    cached = self.client.query_pages(
                        "index_weight",
                        7000,
                        index_code=index.tushare_code,
                        start_date=start,
                        end_date=end,
                    )
                    atomic_parquet(cached, path)
            metadata_path.write_text(
                json.dumps({"index_code": index.tushare_code}, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )

    def download_all(self) -> None:
        self.config.validate()
        marker = self.config.raw_dir / "_SUCCESS.json"
        marker.unlink(missing_ok=True)
        stocks, classes = self._download_reference()
        industry_codes = self._download_industry_members(classes)
        stock_rows = list(stocks.itertuples(index=False))
        progress = tqdm(stock_rows, desc="下载股票", unit="只", dynamic_ncols=True)
        for stock in progress:
            self._download_stock(stock, progress)
        self._download_index_weights()
        marker.parent.mkdir(parents=True, exist_ok=True)
        temporary = marker.with_name(f".{marker.name}.{uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "start_date": self.config.start_date,
                    "end_date": self.config.end_date,
                    "future_calendar_end_date": self.config.future_calendar_end_date,
                    "stock_count": len(stocks),
                    "stocks": stocks["ts_code"].astype(str).tolist(),
                    "industry_codes": industry_codes,
                    "indices": [index.market for index in self.config.indices],
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(marker)


def main() -> None:
    Downloader().download_all()


if __name__ == "__main__":
    main()
