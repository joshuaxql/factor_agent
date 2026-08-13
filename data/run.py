"""Execute configured Tushare download, normalization, and Qlib build stages."""

from __future__ import annotations

from data.config import CONFIG, DataConfig


def _stage(message: str) -> None:
    from tqdm.auto import tqdm

    tqdm.write(message)


def run(config: DataConfig = CONFIG) -> None:
    config.validate()
    if config.run_download:
        from data.download import Downloader

        _stage("[1/3] 下载 Tushare 数据")
        Downloader(config).download_all()
    if config.run_normalize:
        from data.normalize import normalize_all

        _stage("[2/3] 标准化股票数据")
        normalize_all(config)
    if config.run_build_provider:
        from data.provider import build_provider

        _stage("[3/3] 构建 Qlib provider")
        build_provider(config)
        _stage(f"完成：{config.provider_uri.expanduser().resolve()}")
    elif config.run_verify:
        from data.provider import verify_provider

        verify_provider(config.provider_uri, config)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
