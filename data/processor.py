"""Phase 2：原始数据 → 成品日线数据（多进程并行）。

每个交易日独立处理，无 API 调用，适合多进程并行：

  读取 ``cache/raw/{symbols,bars,mktvalue}/<date>.csv``（GM 格式）
  → 合并 → 计算 vwap → 后复权 → 符号转 qlib 格式
  → 写入 ``cache/daily/<date>.csv``

本模块 **不导入 gm.api**，确保 multiprocessing spawn 模式下子进程不会
触发掘金终端连接。
"""
from __future__ import annotations

import os
from multiprocessing import Pool
from typing import Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from data.config import (
    DAILY_COLUMNS,
    DAILY_DIR,
    PRICE_FIELDS,
    RAW_BARS_DIR,
    RAW_MKTVALUE_DIR,
    RAW_SYMBOLS_DIR,
    gm_to_qlib,
)


def _get_dates_with_raw() -> list[str]:
    """返回三份原始文件都齐备的日期列表（按日期升序）。"""
    sym_dates = {f.stem for f in RAW_SYMBOLS_DIR.glob("*.csv")}
    bar_dates = {f.stem for f in RAW_BARS_DIR.glob("*.csv")}
    mv_dates = {f.stem for f in RAW_MKTVALUE_DIR.glob("*.csv")}
    return sorted(sym_dates & bar_dates & mv_dates)


def process_day(trade_date: str) -> bool:
    """处理单个交易日的原始数据，输出成品日线 CSV。

    独立函数（非方法），可被 multiprocessing 直接 pickle。
    """
    try:
        sym_path = RAW_SYMBOLS_DIR / f"{trade_date}.csv"
        bars_path = RAW_BARS_DIR / f"{trade_date}.csv"
        mv_path = RAW_MKTVALUE_DIR / f"{trade_date}.csv"

        if not all(p.exists() for p in (sym_path, bars_path, mv_path)):
            logger.warning(f"{trade_date}: 原始数据不完整，跳过")
            return False

        # ---- 读取原始 CSV（GM 格式符号）----
        df_sym = pd.read_csv(sym_path, dtype={"symbol": str})
        df_bars = pd.read_csv(bars_path, dtype={"symbol": str})
        df_mv = pd.read_csv(mv_path, dtype={"symbol": str})

        if df_sym.empty:
            return False

        # ---- 合并：以 get_symbols 为基准，左连接 bars 和 mktvalue ----
        cols_sym = ["symbol", "upper_limit", "lower_limit", "turn_rate", "adj_factor"]
        cols_sym = [c for c in cols_sym if c in df_sym.columns]
        df = df_sym[cols_sym].copy()

        if not df_bars.empty:
            cols_bars = ["symbol", "open", "high", "low", "close", "volume", "amount"]
            cols_bars = [c for c in cols_bars if c in df_bars.columns]
            df = df.merge(df_bars[cols_bars], on="symbol", how="left")

        if not df_mv.empty and "tot_mv" in df_mv.columns:
            df = df.merge(df_mv[["symbol", "tot_mv"]], on="symbol", how="left")

        # ---- vwap = amount / volume ----
        if "volume" in df.columns and "amount" in df.columns:
            vol = df["volume"]
            df["vwap"] = df["amount"] / vol.where(vol > 0)
        else:
            df["vwap"] = float("nan")

        # ---- 后复权：价格类字段 *= adj_factor ----
        if "adj_factor" in df.columns:
            adj = df["adj_factor"].fillna(1.0)
            for col in PRICE_FIELDS:
                if col in df.columns:
                    df[col] = df[col] * adj
        else:
            df["adj_factor"] = 1.0

        # ---- 符号转 qlib 格式 + 日期 ----
        df["symbol"] = df["symbol"].astype(str).apply(gm_to_qlib)
        df["date"] = trade_date

        # ---- 选取列并保存 ----
        cols = [c for c in DAILY_COLUMNS if c in df.columns]
        df = df[cols]

        DAILY_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DAILY_DIR / f"{trade_date}.csv"
        df.to_csv(out_path, index=False)
        return True

    except Exception as e:  # noqa: BLE001
        logger.error(f"{trade_date}: 处理失败 — {e}")
        return False


def process_all(
    workers: Optional[int] = None,
    resume: bool = True,
) -> None:
    """多进程处理所有已下载的原始数据。

    Args:
        workers: 进程池大小，None = CPU 核数。
        resume: 为 True 时跳过已存在 ``daily/<date>.csv`` 的日期。
    """
    DAILY_DIR.mkdir(parents=True, exist_ok=True)

    dates = _get_dates_with_raw()
    if not dates:
        logger.error("未找到原始数据，请先运行下载阶段 (python -m data.run --phase download)")
        return

    if resume:
        pending = [d for d in dates if not (DAILY_DIR / f"{d}.csv").exists()]
        logger.info(f"处理阶段: 共 {len(dates)} 天已下载，已处理 {len(dates) - len(pending)} 天，剩余 {len(pending)} 天")
    else:
        pending = dates

    if not pending:
        logger.info("无需处理")
        return

    if workers is None:
        workers = os.cpu_count() or 4

    logger.info(f"启动 {workers} 进程处理 {len(pending)} 天数据...")

    # imap_unordered 保证先完成的先返回，进度条更平滑
    with Pool(processes=workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_day, pending),
                total=len(pending),
                desc="处理",
                ncols=80,
            )
        )

    success = sum(results)
    failed = len(pending) - success
    logger.info(f"处理结束 | 成功 {success}/{len(pending)} 天" + (f"，失败 {failed} 天" if failed else ""))
