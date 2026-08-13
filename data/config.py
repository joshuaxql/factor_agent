"""Configuration for the Tushare data pipeline.

Edit this file to change pipeline behavior. The pipeline intentionally has no
command-line parser so that one checked-in configuration defines every build.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)


@dataclass(frozen=True)
class IndexConfig:
    tushare_code: str
    market: str


INDICES = (
    IndexConfig("399300.SZ", "csi300"),
    IndexConfig("000905.SH", "csi500"),
    IndexConfig("000906.SH", "csi800"),
    IndexConfig("000852.SH", "csi1000"),
    IndexConfig("000985.CSI", "csiall"),
)

STOCK_FIELDS = (
    "ts_code,symbol,name,area,industry,market,exchange,curr_type,"
    "list_status,list_date,delist_date"
)
TRADE_CAL_FIELDS = "exchange,cal_date,is_open,pretrade_date"
SW_CLASS_FIELDS = "index_code,industry_name,parent_code,level,industry_code,is_pub,src"

ENDPOINT_FIELDS = {
    "daily": "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
    "adj_factor": "ts_code,trade_date,adj_factor",
    "daily_basic": (
        "ts_code,trade_date,turnover_rate,volume_ratio,pe_ttm,ps_ttm,dv_ttm,"
        "total_mv,circ_mv,limit_status"
    ),
    "index_member_all": (
        "l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,"
        "ts_code,name,in_date,out_date,is_new"
    ),
    "index_weight": "index_code,con_code,trade_date,weight",
}

ENDPOINT_ROW_LIMITS = {
    "daily": 6000,
    "adj_factor": 6000,
    "daily_basic": 6000,
    "index_member_all": 2000,
}

STANDARD_COLUMNS = (
    "symbol",
    "date",
    "industry",
    "industry_code",
    "open",
    "high",
    "low",
    "close",
    "factor",
    "turnover",
    "volume_ratio",
    "pe_ttm",
    "ps_ttm",
    "dv_ttm",
    "total_mv",
    "circ_mv",
    "limit_status",
    "volume",
    "amount",
    "vwap",
    "change",
)

# Qlib feature files are float32. The string industry name remains in the
# standardized Parquet data; $industry contains its numeric SW2021 L1 code.
QLIB_FEATURE_FIELDS = (
    "industry",
    "open",
    "high",
    "low",
    "close",
    "factor",
    "turnover",
    "volume_ratio",
    "pe_ttm",
    "ps_ttm",
    "dv_ttm",
    "total_mv",
    "circ_mv",
    "limit_status",
    "volume",
    "vwap",
)


@dataclass(frozen=True)
class DataConfig:
    start_date: str = "20000101"
    end_date: str = "20260731"
    future_calendar_end_date: str = "20271231"

    data_root: Path = PROJECT_ROOT / ".data" / "tushare"
    provider_uri: Path = Path(
        os.environ.get("QLIB_DATA", "~/.qlib/qlib_data/cn_data")
    ).expanduser()
    tushare_token: str = os.environ.get("TUSHARE_TOKEN", "").strip()

    run_download: bool = True
    run_normalize: bool = True
    run_build_provider: bool = True
    run_verify: bool = True
    resume: bool = True
    replace_existing_provider: bool = True

    stock_exchanges: tuple[str, ...] = ("SSE", "SZSE", "BSE")
    stock_markets: tuple[str, ...] = ("主板", "中小板", "创业板", "科创板", "北交所")
    stock_statuses: tuple[str, ...] = ("L", "D", "P")
    sample_ts_codes: tuple[str, ...] = ()
    indices: tuple[IndexConfig, ...] = INDICES
    download_window_years: int = 15
    request_interval_seconds: float = 0.32
    request_timeout_seconds: int = 30
    max_retries: int = 5
    retry_delay_seconds: float = 1.0
    price_normalization_tolerance: float = 1e-5

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def standard_dir(self) -> Path:
        return self.data_root / "standard"

    def validate(self) -> None:
        try:
            start = datetime.strptime(self.start_date, "%Y%m%d").date()
            end = datetime.strptime(self.end_date, "%Y%m%d").date()
            future_end = datetime.strptime(self.future_calendar_end_date, "%Y%m%d").date()
        except ValueError as exc:
            raise ValueError("start_date, end_date, and future_calendar_end_date must use YYYYMMDD")
        if start > end:
            raise ValueError("start_date must not be after end_date")
        if future_end <= end:
            raise ValueError("future_calendar_end_date must be after end_date")
        if self.run_download and end >= date.today():
            raise ValueError("end_date must be a completed date before today when downloading")
        if self.download_window_years <= 0:
            raise ValueError("download_window_years must be positive")
        if self.request_interval_seconds < 0:
            raise ValueError("request_interval_seconds must not be negative")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.max_retries <= 0:
            raise ValueError("max_retries must be positive")
        if self.sample_ts_codes and self.data_root.resolve() == (PROJECT_ROOT / ".data" / "tushare").resolve():
            raise ValueError("sample_ts_codes must use a separate data_root to avoid polluting full-download state")


CONFIG = DataConfig()
