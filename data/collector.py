"""Phase 1：掘金量化原始数据下载器。

每个交易日的下载流程：
  1. ``stk_get_index_constituents`` ×3  → 成分股（qlib 格式存 ``cache/constituents/``）
  2. 成分股取并集 → ``get_symbols(skip_suspended=True, skip_st=False)``
     → 原始 CSV 存 ``cache/raw/symbols/<date>.csv``
  3. ``history(adjust=ADJUST_NONE)``      → 原始 CSV 存 ``cache/raw/bars/<date>.csv``
  4. ``stk_get_daily_mktvalue_pt``        → 原始 CSV 存 ``cache/raw/mktvalue/<date>.csv``

不做任何合并 / vwap / 后复权 / 符号转换 —— 那是 Phase 2 (``processor.py``) 的职责。
支持线程池跨日并发下载。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from gm.api import (
    ADJUST_NONE,
    set_token,
    get_trading_dates,
    stk_get_index_constituents,
    get_symbols,
    history,
    stk_get_daily_mktvalue_pt,
)

from data.config import (
    BATCH_SIZE,
    CACHE_DIR,
    CONSTITUENTS_DIR,
    DOWNLOAD_WORKERS,
    GM_TOKEN,
    INDICES,
    MAX_RETRY,
    RAW_BARS_DIR,
    RAW_MKTVALUE_DIR,
    RAW_SYMBOLS_DIR,
    RETRY_DELAY,
    gm_to_qlib,
    qlib_to_gm,
)


def _batch(items: list, size: int) -> Iterable[list]:
    """将列表切分为每块 ``size`` 个的子列表。"""
    for i in range(0, len(items), size):
        yield items[i : i + size]


class Downloader:
    """从掘金终端下载原始数据并缓存为 CSV。"""

    def __init__(
        self,
        token: str = GM_TOKEN,
        start_date: str = "2010-01-01",
        end_date: str = "2026-06-30",
        indices: Optional[list[str]] = None,
        batch_size: int = BATCH_SIZE,
        max_retry: int = MAX_RETRY,
        retry_delay: float = RETRY_DELAY,
    ) -> None:
        if not token:
            raise ValueError("GM_TOKEN 未设置，请在 .env 中配置 GM_TOKEN")

        self.token = token
        self.start_date = start_date
        self.end_date = end_date
        self.indices = indices or list(INDICES.keys())
        self.batch_size = batch_size
        self.max_retry = max_retry
        self.retry_delay = retry_delay

        # 创建缓存目录
        for d in (CONSTITUENTS_DIR, RAW_SYMBOLS_DIR, RAW_BARS_DIR, RAW_MKTVALUE_DIR):
            d.mkdir(parents=True, exist_ok=True)

        set_token(token)
        logger.info(
            f"下载器初始化 | 范围 {start_date}~{end_date} | 指数 {self.indices}"
        )

    # ------------------------------------------------------------------ #
    # 基础工具
    # ------------------------------------------------------------------ #

    def _retry_call(self, func: Callable, *args, **kwargs):
        """带指数退避的 API 调用重试。"""
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retry):
            try:
                return func(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                last_err = e
                wait = self.retry_delay * (2**attempt)
                logger.warning(
                    f"API {func.__name__} 第 {attempt + 1}/{self.max_retry} 次失败: {e}，{wait:.1f}s 后重试"
                )
                time.sleep(wait)
        assert last_err is not None
        raise last_err

    def _get_trading_dates(self) -> list[str]:
        """获取上交所交易日历 [start_date, end_date]，缓存为 CSV。"""
        raw = self._retry_call(
            get_trading_dates, "SHSE", self.start_date, self.end_date
        )
        dates: list[str] = []
        for d in raw:
            if isinstance(d, str):
                dates.append(d[:10])
            else:
                dates.append(d.strftime("%Y-%m-%d"))
        if not dates:
            logger.warning("交易日历为空，请检查日期范围")
            return []
        logger.info(f"交易日历: {len(dates)} 天，{dates[0]} ~ {dates[-1]}")

        path = CACHE_DIR / "trading_dates.csv"
        pd.DataFrame({"date": dates}).to_csv(path, index=False)
        return dates

    # ------------------------------------------------------------------ #
    # 单个 API 封装（返回原始 DataFrame，不做任何处理）
    # ------------------------------------------------------------------ #

    def _fetch_constituents(self, index: str, trade_date: str) -> pd.DataFrame:
        """``stk_get_index_constituents`` → DataFrame（含 GM 格式 symbol 列）。"""
        df = self._retry_call(
            stk_get_index_constituents, index=index, trade_date=trade_date
        )
        if df is None or len(df) == 0:
            return pd.DataFrame()
        return df

    def _fetch_symbols(self, symbols: list[str], trade_date: str) -> pd.DataFrame:
        """``get_symbols`` → DataFrame（GM 格式 symbol，含 adj_factor/upper_limit 等）。"""
        if not symbols:
            return pd.DataFrame()
        results: list[pd.DataFrame] = []
        for batch in _batch(symbols, self.batch_size):
            sym_str = ",".join(batch)
            df = self._retry_call(
                get_symbols,
                sec_type1=1010,
                symbols=sym_str,
                skip_suspended=True,
                skip_st=False,
                trade_date=trade_date,
                df=True,
            )
            if df is not None and len(df) > 0:
                results.append(df)
        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)

    def _fetch_bars(self, symbols: list[str], trade_date: str) -> pd.DataFrame:
        """``history(ADJUST_NONE)`` → DataFrame（GM 格式 symbol + OHLCV）。"""
        if not symbols:
            return pd.DataFrame()
        start_time = f"{trade_date} 00:00:00"
        end_time = f"{trade_date} 23:59:59"
        results: list[pd.DataFrame] = []
        for batch in _batch(symbols, self.batch_size):
            sym_str = ",".join(batch)
            df = self._retry_call(
                history,
                symbol=sym_str,
                frequency="1d",
                start_time=start_time,
                end_time=end_time,
                fields="symbol,open,high,low,close,volume,amount",
                adjust=ADJUST_NONE,
                df=True,
            )
            if df is not None and len(df) > 0:
                results.append(df)
        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)

    def _fetch_mktvalue(self, symbols: list[str], trade_date: str) -> pd.DataFrame:
        """``stk_get_daily_mktvalue_pt`` → DataFrame（GM 格式 symbol + tot_mv）。"""
        if not symbols:
            return pd.DataFrame()
        results: list[pd.DataFrame] = []
        for batch in _batch(symbols, self.batch_size):
            sym_str = ",".join(batch)
            df = self._retry_call(
                stk_get_daily_mktvalue_pt,
                symbols=sym_str,
                fields="tot_mv",
                trade_date=trade_date,
                df=True,
            )
            if df is not None and len(df) > 0:
                results.append(df)
        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)

    # ------------------------------------------------------------------ #
    # 单日下载
    # ------------------------------------------------------------------ #

    @staticmethod
    def _raw_complete(trade_date: str) -> bool:
        """检查某日的三份原始文件是否都已存在。"""
        return all(
            (d / f"{trade_date}.csv").exists()
            for d in (RAW_SYMBOLS_DIR, RAW_BARS_DIR, RAW_MKTVALUE_DIR)
        )

    def download_day(self, trade_date: str) -> bool:
        """下载单个交易日的全部原始数据。

        每一步都跳过已存在的文件，支持断点续传。
        """
        try:
            # ---- 1. 成分股（qlib 格式存盘，便于后续直接使用）----
            all_gm_symbols: set[str] = set()
            for index in self.indices:
                index_qlib = gm_to_qlib(index)
                const_dir = CONSTITUENTS_DIR / index_qlib
                const_dir.mkdir(parents=True, exist_ok=True)
                const_path = const_dir / f"{trade_date}.csv"

                if const_path.exists():
                    df_const = pd.read_csv(const_path, dtype={"symbol": str})
                else:
                    df_const = self._fetch_constituents(index, trade_date)
                    if not df_const.empty:
                        df_const = df_const.copy()
                        df_const["symbol"] = (
                            df_const["symbol"].astype(str).apply(gm_to_qlib)
                        )
                        df_const.to_csv(const_path, index=False)

                if not df_const.empty and "symbol" in df_const.columns:
                    all_gm_symbols.update(
                        qlib_to_gm(s) for s in df_const["symbol"].tolist()
                    )

            if not all_gm_symbols:
                logger.warning(f"{trade_date}: 无成分股数据，跳过")
                return False

            gm_symbols = sorted(all_gm_symbols)

            # ---- 2. get_symbols → raw/symbols/<date>.csv ----
            sym_path = RAW_SYMBOLS_DIR / f"{trade_date}.csv"
            if not sym_path.exists():
                df_sym = self._fetch_symbols(gm_symbols, trade_date)
                if df_sym.empty:
                    logger.warning(f"{trade_date}: get_symbols 返回空，跳过")
                    return False
                df_sym.to_csv(sym_path, index=False)
            else:
                df_sym = pd.read_csv(sym_path, dtype={"symbol": str})

            active_symbols = df_sym["symbol"].tolist()

            # ---- 3. history → raw/bars/<date>.csv ----
            bars_path = RAW_BARS_DIR / f"{trade_date}.csv"
            if not bars_path.exists():
                df_bars = self._fetch_bars(active_symbols, trade_date)
                if not df_bars.empty:
                    df_bars.to_csv(bars_path, index=False)

            # ---- 4. mktvalue → raw/mktvalue/<date>.csv ----
            mv_path = RAW_MKTVALUE_DIR / f"{trade_date}.csv"
            if not mv_path.exists():
                df_mv = self._fetch_mktvalue(active_symbols, trade_date)
                if not df_mv.empty:
                    df_mv.to_csv(mv_path, index=False)

            # logger.info(f"{trade_date}: 下载完成 ({len(active_symbols)} 只股票)")
            return True

        except Exception as e:  # noqa: BLE001
            logger.error(f"{trade_date}: 下载失败 — {e}")
            return False

    # ------------------------------------------------------------------ #
    # 批量下载
    # ------------------------------------------------------------------ #

    def download_all(
        self, resume: bool = True, workers: int = DOWNLOAD_WORKERS
    ) -> None:
        """遍历交易日历，逐日下载原始数据。

        Args:
            resume: 为 True 时跳过三份原始文件都已存在的交易日。
            workers: 线程池大小（跨日并发），1 为单线程。
        """
        dates = self._get_trading_dates()
        if not dates:
            return

        # 缓存交易日列表
        pd.DataFrame({"date": dates}).to_csv(
            CACHE_DIR / "trading_dates.csv", index=False
        )

        if resume:
            pending = [d for d in dates if not self._raw_complete(d)]
            logger.info(
                f"断点续传: 已完成 {len(dates) - len(pending)} 天，剩余 {len(pending)} 天"
            )
        else:
            pending = dates

        if not pending:
            logger.info("无需下载")
            return

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(self.download_day, d): d for d in pending}
                for future in tqdm(
                    as_completed(futures), total=len(pending), desc="下载", ncols=80
                ):
                    futures[future]  # 触发异常（download_day 内部已捕获）
        else:
            for date in tqdm(pending, desc="下载", ncols=80):
                self.download_day(date)

        success = sum(1 for d in pending if self._raw_complete(d))
        logger.info(f"下载结束 | 成功 {success}/{len(pending)} 天")
