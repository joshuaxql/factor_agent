"""将按日缓存的 CSV 透视为按股票的 CSV。

按日缓存（``cache/daily/<date>.csv``）适合采集阶段的原子写入，
但 qlib bin 格式和因子计算需要按股票组织数据。
本脚本将所有按日 CSV 合并后按 ``symbol`` 分组输出到 ``cache/merged/<symbol>.csv``。

用法::

    python -m data.merge                # 全量合并
    python -m data.merge --workers 8    # 多进程加速
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from loguru import logger
from tqdm import tqdm

from data.config import DAILY_COLUMNS, DAILY_DIR, MERGED_DIR


def merge_all(workers: int = 1) -> None:
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(DAILY_DIR.glob("*.csv"))
    if not csv_files:
        logger.error(f"未找到日线缓存文件 ({DAILY_DIR})，请先运行采集")
        return
    logger.info(f"读取 {len(csv_files)} 个按日 CSV 文件...")

    # 一次性读取并拼接
    df = pd.concat(
        (pd.read_csv(f, dtype={"symbol": str}) for f in csv_files),
        ignore_index=True,
    )
    logger.info(f"合计 {len(df):,} 行，{df['symbol'].nunique()} 只股票")

    # 按日期排序
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # 统一列顺序
    cols = [c for c in DAILY_COLUMNS if c in df.columns]
    df = df[cols]

    # 按股票分组写出
    logger.info(f"写出按股票 CSV 到 {MERGED_DIR}...")
    for symbol, group in tqdm(df.groupby("symbol"), desc="合并", ncols=80):
        out_path = MERGED_DIR / f"{symbol}.csv"
        group.to_csv(out_path, index=False)

    logger.info(f"合并完成: {df['symbol'].nunique()} 个文件 → {MERGED_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="按日 CSV → 按股票 CSV")
    parser.add_argument("--workers", type=int, default=1, help="并发数（暂未使用）")
    args = parser.parse_args()
    merge_all(workers=args.workers)


if __name__ == "__main__":
    main()
