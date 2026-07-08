"""采集配置、路径与符号转换工具。

所有路径基于 ``QLIB_DATA`` 环境变量（默认 ``~/.qlib/qlib_data/cn_data``）。
缓存分两层：

- **原始层** (``cache/raw/``)：Phase 1 直接保存 API 返回值（GM 格式符号）
- **成品层** (``cache/daily/``、``cache/constituents/``)：Phase 2 处理后输出（qlib 格式符号）
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# 数据范围
# --------------------------------------------------------------------------- #
START_DATE = "2010-01-01"
END_DATE = "2026-06-30"

# --------------------------------------------------------------------------- #
# 指数成分 —— 每个交易日通过 stk_get_index_constituents 获取
# --------------------------------------------------------------------------- #
INDICES: dict[str, str] = {
    "SHSE.000300": "沪深300",
    "SHSE.000905": "中证500",
    "SHSE.000906": "中证800",
}

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
QLIB_DATA = Path(os.environ.get("QLIB_DATA", "~/.qlib/qlib_data/cn_data")).expanduser()
CACHE_DIR = QLIB_DATA / "cache"

# 成品层
CONSTITUENTS_DIR = CACHE_DIR / "constituents"  # 指数成分股（qlib 格式）
DAILY_DIR = CACHE_DIR / "daily"  # 按交易日的日线数据（qlib 格式）
MERGED_DIR = CACHE_DIR / "merged"  # 按股票合并后的日线数据

# 原始层（Phase 1 下载，GM 格式符号，不做任何处理）
RAW_DIR = CACHE_DIR / "raw"
RAW_SYMBOLS_DIR = RAW_DIR / "symbols"  # get_symbols 输出
RAW_BARS_DIR = RAW_DIR / "bars"  # history 输出
RAW_MKTVALUE_DIR = RAW_DIR / "mktvalue"  # stk_get_daily_mktvalue_pt 输出

# --------------------------------------------------------------------------- #
# 掘金 token
# --------------------------------------------------------------------------- #
GM_TOKEN = os.environ.get("GM_TOKEN", "")

# --------------------------------------------------------------------------- #
# 日线数据列定义
# --------------------------------------------------------------------------- #
# 最终保存的日线数据列（顺序即 CSV 列顺序）
DAILY_COLUMNS: list[str] = [
    "symbol",  # qlib 格式，如 sh600000
    "date",  # 交易日，如 2024-01-15
    "open",  # 后复权
    "high",  # 后复权
    "low",  # 后复权
    "close",  # 后复权
    "vwap",  # 后复权，由 amount/volume 计算
    "volume",  # 原始（不复权）
    "amount",  # 原始（不复权）
    "turn_rate",  # 换手率 %
    "upper_limit",  # 后复权
    "lower_limit",  # 后复权
    "tot_mv",  # 总市值（元）
    "adj_factor",  # 后复权因子（参考用）
]

# 需要乘以 adj_factor 做后复权的价格类字段
PRICE_FIELDS: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "upper_limit",
    "lower_limit",
]

# --------------------------------------------------------------------------- #
# 采集参数
# --------------------------------------------------------------------------- #
# 单日查询 ~800 只股票，行数远低于 history 的 33000 条上限，可用大 batch
BATCH_SIZE = 1000

MAX_RETRY = 3  # API 调用失败重试次数
RETRY_DELAY = 1.0  # 重试初始延迟（秒），指数退避

DOWNLOAD_WORKERS = 1  # Phase 1 线程池大小（跨日并发下载）
PROCESS_WORKERS = None  # Phase 2 进程池大小（None = CPU 核数）

# --------------------------------------------------------------------------- #
# 符号转换：掘金格式 (SHSE.600000) <-> qlib 格式 (sh600000)
# --------------------------------------------------------------------------- #
_GM_PREFIX: dict[str, str] = {"SHSE": "sh", "SZSE": "sz"}
_QLIB_PREFIX: dict[str, str] = {v: k for k, v in _GM_PREFIX.items()}


def gm_to_qlib(gm_symbol: str) -> str:
    """SHSE.600000 -> sh600000 / SZSE.000001 -> sz000001"""
    exchange, code = gm_symbol.split(".")
    prefix = _GM_PREFIX.get(exchange)
    if prefix is None:
        raise ValueError(f"不支持的交易所代码: {exchange}（符号 {gm_symbol}）")
    return prefix + code


def qlib_to_gm(qlib_symbol: str) -> str:
    """sh600000 -> SHSE.600000 / sz000001 -> SZSE.000001"""
    prefix = qlib_symbol[:2]
    exchange = _QLIB_PREFIX.get(prefix)
    if exchange is None:
        raise ValueError(f"不支持的 qlib 前缀: {prefix}（符号 {qlib_symbol}）")
    return f"{exchange}.{qlib_symbol[2:]}"
