"""One-click GM data build pipeline.

Pipeline:

1. Download raw GM data into ``<QLIB_DATA>/cache/raw``.
2. Process raw files into daily qlib-style CSV under ``<QLIB_DATA>/cache/daily``.
3. Dump daily CSV files into qlib binary provider files under ``<QLIB_DATA>``.

Examples::

    python -m data.build_qlib
    python -m data.build_qlib --provider-uri D:\\data\\qlib --start 2010-01-01 --end 2026-06-30
    python -m data.build_qlib --skip-download --skip-process
"""

from __future__ import annotations

import argparse
import io
import sys

from data.config import QLIB_DATA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键下载、处理并转换掘金数据为 qlib 格式")
    parser.add_argument("--provider-uri", default=str(QLIB_DATA), help="qlib provider 输出目录，默认读取 QLIB_DATA")
    parser.add_argument("--start", default="2010-01-01", help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-06-30", help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="断点续传/跳过已完成文件")
    parser.add_argument("--download-workers", type=int, default=4, help="下载阶段线程池大小")
    parser.add_argument("--process-workers", type=int, default=None, help="处理阶段进程池大小（默认 CPU 核数）")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载阶段，只使用现有 cache/raw")
    parser.add_argument("--skip-process", action="store_true", help="跳过处理阶段，只使用现有 cache/daily")
    parser.add_argument("--skip-dump", action="store_true", help="跳过 qlib bin 转换阶段")
    parser.add_argument("--clear-existing", action="store_true", help="dump 前清空 calendars/instruments/features")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True, help="dump 后检查 provider 结构")
    parser.add_argument("--sample-instruments", type=int, default=0, help="仅 dump 前 N 只股票，用于调试")
    parser.add_argument("--include-aliases", action="store_true", help="额外导出 amount->money、turn_rate->turn/turnover 兼容别名")
    return parser.parse_args()


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    args = parse_args()

    if not args.skip_download:
        from data.collector import Downloader

        downloader = Downloader(start_date=args.start, end_date=args.end)
        downloader.download_all(resume=args.resume, workers=args.download_workers)

    if not args.skip_process:
        from data.processor import process_all

        process_all(workers=args.process_workers, resume=args.resume)

    if not args.skip_dump:
        from data.dump_bin import dump_qlib_data

        dump_qlib_data(
            provider_uri=args.provider_uri,
            start=args.start,
            end=args.end,
            sample_instruments=args.sample_instruments,
            include_aliases=args.include_aliases,
            clear_existing=args.clear_existing,
            verify=args.verify,
        )


if __name__ == "__main__":
    main()
