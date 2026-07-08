"""CLI 入口：掘金数据采集与 qlib 格式转换（三阶段管线）。

用法::

    # 全流程（下载 + 处理 + dump qlib bin）
    python -m data.run

    # 仅下载原始数据（线程池并发）
    python -m data.run --phase download --download-workers 4

    # 仅处理（多进程并行）
    python -m data.run --phase process --process-workers 8

    # 指定日期范围
    python -m data.run --start 2020-01-01 --end 2024-12-31

    # 强制重新处理（覆盖已有成品）
    python -m data.run --phase process --no-resume

    # 仅将现有 cache/daily 转换为 qlib provider 格式
    python -m data.run --phase dump --provider-uri D:\\data\\qlib
"""
from __future__ import annotations

import argparse
import io
import sys

# Windows 终端 UTF-8 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="掘金量化日线数据采集与 qlib 格式转换")
    parser.add_argument("--phase", choices=["download", "process", "dump", "all"], default="all",
                        help="执行阶段：download=仅下载, process=仅处理, dump=仅转换qlib格式, all=全部")
    parser.add_argument("--start", default="2010-01-01", help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-06-30", help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="断点续传（跳过已完成的日期）")
    parser.add_argument("--download-workers", type=int, default=4,
                        help="下载阶段线程池大小（跨日并发）")
    parser.add_argument("--process-workers", type=int, default=None,
                        help="处理阶段进程池大小（默认 CPU 核数）")
    parser.add_argument("--provider-uri", default=None,
                        help="qlib provider 输出目录（默认读取 QLIB_DATA）")
    parser.add_argument("--clear-existing", action="store_true",
                        help="dump 前清空 calendars/instruments/features")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True,
                        help="dump 后检查 provider 结构")
    parser.add_argument("--sample-instruments", type=int, default=0,
                        help="仅 dump 前 N 只股票，用于调试")
    parser.add_argument("--include-aliases", action="store_true",
                        help="额外导出 amount->money、turn_rate->turn/turnover 兼容别名")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.phase in ("download", "all"):
        from data.collector import Downloader

        downloader = Downloader(start_date=args.start, end_date=args.end)
        downloader.download_all(resume=args.resume, workers=args.download_workers)

    if args.phase in ("process", "all"):
        from data.processor import process_all

        process_all(workers=args.process_workers, resume=args.resume)

    if args.phase in ("dump", "all"):
        from data.config import QLIB_DATA
        from data.dump_bin import dump_qlib_data

        dump_qlib_data(
            provider_uri=args.provider_uri or QLIB_DATA,
            start=args.start,
            end=args.end,
            sample_instruments=args.sample_instruments,
            include_aliases=args.include_aliases,
            clear_existing=args.clear_existing,
            verify=args.verify,
        )


if __name__ == "__main__":
    main()
