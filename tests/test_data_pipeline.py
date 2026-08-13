from __future__ import annotations

import tempfile
import unittest
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from data.config import ENDPOINT_FIELDS, INDICES, DataConfig, IndexConfig
from data.download import Downloader, atomic_parquet, date_windows, month_windows
from data.normalize import normalize_all
from data.provider import (
    _publish,
    _read_bin,
    build_provider,
    refresh_provider_symbols,
    verify_provider,
)


class DataPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.config = DataConfig(
            start_date="20200102",
            end_date="20200203",
            future_calendar_end_date="20200210",
            data_root=root / "data",
            provider_uri=root / "provider",
            run_download=False,
            run_normalize=True,
            run_build_provider=True,
            run_verify=True,
            resume=False,
        )
        self._write_fixtures()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, relative: str, frame: pd.DataFrame) -> None:
        atomic_parquet(frame, self.config.raw_dir / relative)

    def _write_fixtures(self) -> None:
        stock = {
            "ts_code": "000001.SZ",
            "symbol": "000001",
            "name": "Test",
            "area": "Test",
            "industry": "Test",
            "market": "主板",
            "exchange": "SZSE",
            "curr_type": "CNY",
            "list_status": "L",
            "list_date": "20100101",
            "delist_date": None,
        }
        self._write("reference/stock_basic_L.parquet", pd.DataFrame([stock]))
        self._write("reference/stock_universe.parquet", pd.DataFrame([stock]))
        self._write(
            "reference/trade_cal.parquet",
            pd.DataFrame(
                {
                    "exchange": ["SSE"] * 3,
                    "cal_date": ["20200102", "20200103", "20200203"],
                    "is_open": ["1"] * 3,
                    "pretrade_date": ["20191231", "20200102", "20200123"],
                }
            ),
        )
        self._write(
            "reference/trade_cal_future.parquet",
            pd.DataFrame(
                {
                    "exchange": ["SSE"] * 4,
                    "cal_date": ["20200102", "20200103", "20200203", "20200204"],
                    "is_open": ["1"] * 4,
                    "pretrade_date": ["20191231", "20200102", "20200123", "20200203"],
                }
            ),
        )
        self._write(
            "reference/sw2021_l1.parquet",
            pd.DataFrame(
                {
                    "index_code": ["801010.SI", "801020.SI"],
                    "industry_name": ["Agriculture", "Mining"],
                    "parent_code": ["0", "0"],
                    "level": ["L1", "L1"],
                    "industry_code": ["110000", "210000"],
                    "is_pub": ["1", "1"],
                    "src": ["SW2021", "SW2021"],
                }
            ),
        )

        dates = ["20200102", "20200103", "20200203"]
        daily = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"] * 3,
                "trade_date": dates,
                "open": [9.0, 5.5, 7.0],
                "high": [11.0, 6.0, 7.5],
                "low": [9.0, 5.0, 6.5],
                "close": [10.0, 6.0, 7.0],
                "pre_close": [10.0, 5.0, 6.0],
                "change": [0.0, 1.0, 1.0],
                "pct_chg": [0.0, 20.0, 16.6667],
                "vol": [100.0, 200.0, 300.0],
                "amount": [100.0, 120.0, 210.0],
            }
        )
        self._write("stocks/daily/000001.SZ/chunk.parquet", daily)
        self._write(
            "stocks/adj_factor/000001.SZ/chunk.parquet",
            pd.DataFrame(
                {"ts_code": ["000001.SZ"] * 3, "trade_date": dates, "adj_factor": [1.0, 2.0, 2.0]}
            ),
        )
        basics = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"] * 3,
                "trade_date": dates,
                "turnover_rate": [1.0, 2.0, 3.0],
                "volume_ratio": [1.1, 1.2, 1.3],
                "pe_ttm": [10.0, 11.0, 12.0],
                "ps_ttm": [2.0, 2.1, 2.2],
                "dv_ttm": [1.0, 1.1, 1.2],
                "total_mv": [1000.0, 1200.0, 1400.0],
                "circ_mv": [800.0, 900.0, 1000.0],
                "limit_status": [0, 2, 6],
            }
        )
        self._write("stocks/daily_basic/000001.SZ/chunk.parquet", basics)
        member_columns = ENDPOINT_FIELDS["index_member_all"].split(",")
        members = pd.DataFrame(
            [
                ["801010.SI", "Agriculture", "", "", "", "", "000001.SZ", "Test", "20190101", "20200131", "N"],
                ["801020.SI", "Mining", "", "", "", "", "000001.SZ", "Test", "20200201", None, "Y"],
            ],
            columns=member_columns,
        )
        self._write(
            "index_member_all/801010.SI/historical.parquet",
            members.iloc[[0]].copy(),
        )
        self._write(
            "index_member_all/801010.SI/current.parquet",
            pd.DataFrame(columns=member_columns),
        )
        self._write(
            "index_member_all/801020.SI/historical.parquet",
            pd.DataFrame(columns=member_columns),
        )
        self._write(
            "index_member_all/801020.SI/current.parquet",
            members.iloc[[1]].copy(),
        )

        for index in INDICES:
            self._write(
                f"index_weight/{index.market}/202001.parquet",
                pd.DataFrame(
                    {
                        "index_code": [index.tushare_code],
                        "con_code": ["000001.SZ"],
                        "trade_date": ["20200102"],
                        "weight": [100.0],
                    }
                ),
            )
        marker = self.config.raw_dir / "_SUCCESS.json"
        marker.write_text(
            json.dumps(
                {
                    "start_date": self.config.start_date,
                    "end_date": self.config.end_date,
                    "future_calendar_end_date": self.config.future_calendar_end_date,
                    "stock_count": 1,
                    "stocks": ["000001.SZ"],
                    "industry_codes": ["801010.SI", "801020.SI"],
                    "indices": [index.market for index in self.config.indices],
                }
            ),
            encoding="utf-8",
        )

    def test_window_boundaries(self) -> None:
        self.assertEqual(
            list(date_windows("20000101", "20260731", 15)),
            [("20000101", "20141231"), ("20150101", "20260731")],
        )

    def test_future_calendar_download_extends_current_calendar(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.calls = []

            def request(self, endpoint: str, fields: str, **parameters) -> pd.DataFrame:
                self.calls.append(parameters)
                dates = ["20200102", "20200103", "20200203"]
                if parameters["end_date"] == self_config.future_calendar_end_date:
                    dates.append("20200204")
                return pd.DataFrame(
                    {
                        "exchange": ["SSE"] * len(dates),
                        "cal_date": dates,
                        "is_open": ["1"] * len(dates),
                        "pretrade_date": [""] * len(dates),
                    }
                )

        self_config = replace(self.config, resume=False)
        downloader = object.__new__(Downloader)
        downloader.config = self_config
        downloader.client = Client()

        current, future = downloader._download_trade_calendars(self_config.raw_dir / "calendar_test")

        self.assertEqual(current["cal_date"].tolist(), ["20200102", "20200103", "20200203"])
        self.assertEqual(future["cal_date"].iloc[-1], "20200204")
        self.assertEqual([call["end_date"] for call in downloader.client.calls], ["20200203", "20200210"])

    def test_download_progress_shows_endpoint_and_window(self) -> None:
        class Progress:
            def __init__(self) -> None:
                self.messages = []

            def set_postfix_str(self, message: str, refresh: bool = True) -> None:
                self.messages.append(message)

        class Client:
            def query(self, endpoint: str, **parameters) -> pd.DataFrame:
                return pd.DataFrame(columns=ENDPOINT_FIELDS[endpoint].split(","))

        downloader = object.__new__(Downloader)
        downloader.config = self.config
        downloader.client = Client()
        progress = Progress()
        stock = pd.Series(
            {"ts_code": "000001.SZ", "list_date": "20200102", "delist_date": None}
        )

        downloader._download_stock(stock, progress)

        self.assertTrue(any("000001.SZ daily 20200102-20200203" in value for value in progress.messages))
        self.assertTrue(any("000001.SZ adj_factor" in value for value in progress.messages))
        self.assertTrue(any("000001.SZ daily_basic" in value for value in progress.messages))
        self.assertFalse(any("index_member_all" in value for value in progress.messages))

    def test_incomplete_adjustment_factor_cache_is_refetched(self) -> None:
        window = "20200102_20200203.parquet"
        daily_path = self.config.raw_dir / "stocks" / "daily" / "000001.SZ" / window
        factor_path = self.config.raw_dir / "stocks" / "adj_factor" / "000001.SZ" / window
        basic_path = self.config.raw_dir / "stocks" / "daily_basic" / "000001.SZ" / window
        self._write(
            f"stocks/daily/000001.SZ/{window}",
            pd.read_parquet(self.config.raw_dir / "stocks/daily/000001.SZ/chunk.parquet"),
        )
        self._write(
            f"stocks/adj_factor/000001.SZ/{window}",
            pd.DataFrame(columns=ENDPOINT_FIELDS["adj_factor"].split(",")),
        )
        self._write(
            f"stocks/daily_basic/000001.SZ/{window}",
            pd.read_parquet(self.config.raw_dir / "stocks/daily_basic/000001.SZ/chunk.parquet"),
        )

        class Client:
            def __init__(self) -> None:
                self.calls = []

            def query(self, endpoint: str, **parameters) -> pd.DataFrame:
                self.calls.append(endpoint)
                return pd.DataFrame(
                    {
                        "ts_code": ["000001.SZ"] * 3,
                        "trade_date": ["20200102", "20200103", "20200203"],
                        "adj_factor": [1.0, 2.0, 2.0],
                    }
                )

        downloader = object.__new__(Downloader)
        downloader.config = replace(self.config, resume=True)
        downloader.client = Client()
        stock = pd.Series(
            {"ts_code": "000001.SZ", "list_date": "20200102", "delist_date": None}
        )

        downloader._download_stock(stock)

        self.assertEqual(downloader.client.calls, ["adj_factor"])
        self.assertEqual(len(pd.read_parquet(daily_path)), 3)
        self.assertEqual(len(pd.read_parquet(factor_path)), 3)
        self.assertEqual(len(pd.read_parquet(basic_path)), 3)

    def test_empty_daily_cache_is_refetched_from_basic_dates(self) -> None:
        window = "20200102_20200203.parquet"
        daily_path = self.config.raw_dir / "stocks" / "daily" / "000001.SZ" / window
        factor_path = self.config.raw_dir / "stocks" / "adj_factor" / "000001.SZ" / window
        basic_path = self.config.raw_dir / "stocks" / "daily_basic" / "000001.SZ" / window
        daily = pd.read_parquet(self.config.raw_dir / "stocks/daily/000001.SZ/chunk.parquet")
        factors = pd.read_parquet(
            self.config.raw_dir / "stocks/adj_factor/000001.SZ/chunk.parquet"
        )
        basics = pd.read_parquet(
            self.config.raw_dir / "stocks/daily_basic/000001.SZ/chunk.parquet"
        )
        self._write(
            f"stocks/daily/000001.SZ/{window}",
            pd.DataFrame(columns=ENDPOINT_FIELDS["daily"].split(",")),
        )
        self._write(f"stocks/adj_factor/000001.SZ/{window}", factors)
        self._write(f"stocks/daily_basic/000001.SZ/{window}", basics)

        class Client:
            def __init__(self) -> None:
                self.calls = []

            def query(self, endpoint: str, **parameters) -> pd.DataFrame:
                self.calls.append(endpoint)
                return daily

        downloader = object.__new__(Downloader)
        downloader.config = replace(self.config, resume=True)
        downloader.client = Client()
        stock = pd.Series(
            {"ts_code": "000001.SZ", "list_date": "20200102", "delist_date": None}
        )

        downloader._download_stock(stock)

        self.assertEqual(downloader.client.calls, ["daily"])
        self.assertEqual(len(pd.read_parquet(daily_path)), 3)
        self.assertEqual(len(pd.read_parquet(factor_path)), 3)
        self.assertEqual(len(pd.read_parquet(basic_path)), 3)

    def test_empty_daily_basic_cache_is_refetched_from_daily_dates(self) -> None:
        window = "20200102_20200203.parquet"
        daily = pd.read_parquet(self.config.raw_dir / "stocks/daily/000001.SZ/chunk.parquet")
        factors = pd.read_parquet(
            self.config.raw_dir / "stocks/adj_factor/000001.SZ/chunk.parquet"
        )
        basics = pd.read_parquet(
            self.config.raw_dir / "stocks/daily_basic/000001.SZ/chunk.parquet"
        )
        self._write(f"stocks/daily/000001.SZ/{window}", daily)
        self._write(f"stocks/adj_factor/000001.SZ/{window}", factors)
        self._write(
            f"stocks/daily_basic/000001.SZ/{window}",
            pd.DataFrame(columns=ENDPOINT_FIELDS["daily_basic"].split(",")),
        )

        class Client:
            def __init__(self) -> None:
                self.calls = []

            def query(self, endpoint: str, **parameters) -> pd.DataFrame:
                self.calls.append(endpoint)
                return basics

        downloader = object.__new__(Downloader)
        downloader.config = replace(self.config, resume=True)
        downloader.client = Client()
        stock = pd.Series(
            {"ts_code": "000001.SZ", "list_date": "20200102", "delist_date": None}
        )

        downloader._download_stock(stock)

        self.assertEqual(downloader.client.calls, ["daily_basic"])
        repaired = pd.read_parquet(
            self.config.raw_dir / "stocks/daily_basic/000001.SZ" / window
        )
        self.assertEqual(len(repaired), 3)

    def test_industry_members_download_by_l1_code(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.calls = []

            def query_pages(self, endpoint: str, page_size: int, **parameters) -> pd.DataFrame:
                self.calls.append((endpoint, page_size, parameters))
                return pd.DataFrame(columns=ENDPOINT_FIELDS[endpoint].split(","))

        downloader = object.__new__(Downloader)
        downloader.config = self.config
        downloader.client = Client()
        classes = pd.DataFrame({"index_code": ["801020.SI", "801010.SI"]})

        codes = downloader._download_industry_members(classes)

        self.assertEqual(codes, ["801010.SI", "801020.SI"])
        self.assertEqual(len(downloader.client.calls), 4)
        for endpoint, page_size, parameters in downloader.client.calls:
            self.assertEqual(endpoint, "index_member_all")
            self.assertEqual(page_size, 2000)
            self.assertIn(parameters["l1_code"], codes)
            self.assertIn(parameters["is_new"], {"N", "Y"})
            self.assertNotIn("ts_code", parameters)

    def test_index_weight_cache_tracks_tushare_code(self) -> None:
        index = IndexConfig("000985.CSI", "csiall")
        config = replace(self.config, resume=True, indices=(index,))
        expected = ENDPOINT_FIELDS["index_weight"].split(",")
        for month in ("202001", "202002"):
            self._write(
                f"index_weight/csiall/{month}.parquet",
                pd.DataFrame(columns=expected),
            )

        class Client:
            def __init__(self) -> None:
                self.calls = []

            def query_pages(self, endpoint: str, page_size: int, **parameters) -> pd.DataFrame:
                self.calls.append(parameters)
                return pd.DataFrame(
                    [[index.tushare_code, "000001.SZ", parameters["end_date"], 100.0]],
                    columns=expected,
                )

        downloader = object.__new__(Downloader)
        downloader.config = config
        downloader.client = Client()

        downloader._download_index_weights()

        self.assertEqual(len(downloader.client.calls), 2)
        self.assertTrue(
            all(call["index_code"] == "000985.CSI" for call in downloader.client.calls)
        )
        metadata = json.loads(
            (config.raw_dir / "index_weight/csiall/_SOURCE.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata, {"index_code": "000985.CSI"})

    def test_publish_rejects_regular_file(self) -> None:
        root = Path(self.temporary.name)
        stage = root / "stage"
        stage.mkdir()
        target = root / "target"
        target.write_text("keep", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "non-directory"):
            _publish(stage, target, True, self.config.data_root)

        self.assertEqual(target.read_text(encoding="utf-8"), "keep")
        self.assertTrue(stage.is_dir())
        self.assertEqual(
            list(month_windows("20200115", "20200302")),
            [
                ("202001", "20200115", "20200131"),
                ("202002", "20200201", "20200229"),
                ("202003", "20200301", "20200302"),
            ],
        )

    def test_normalize_and_build_provider(self) -> None:
        normalize_all(self.config)
        standard = pd.read_parquet(
            self.config.standard_dir / "stocks" / "SZ000001.parquet"
        )
        np.testing.assert_allclose(standard["close"], [1.0, 1.2, 1.4])
        np.testing.assert_allclose(standard["factor"], [0.1, 0.2, 0.2])
        np.testing.assert_allclose(standard["close"] / standard["factor"], [10.0, 6.0, 7.0])
        np.testing.assert_allclose(standard["volume"], [1000.0, 1000.0, 1500.0])
        np.testing.assert_allclose(standard["amount"], [100.0, 120.0, 210.0])
        np.testing.assert_allclose(standard["change"].iloc[1:], [-0.4, 1.0 / 6.0])
        np.testing.assert_allclose(standard["limit_status"], [0.0, 2.0, 6.0])
        self.assertEqual(standard["industry"].tolist(), ["Agriculture", "Agriculture", "Mining"])
        np.testing.assert_allclose(standard["industry_code"], [801010.0, 801010.0, 801020.0])

        provider = build_provider(self.config)
        verify_provider(provider, self.config)
        self.assertEqual(
            (provider / "calendars" / "day_future.txt").read_text(encoding="utf-8").splitlines(),
            ["2020-01-02", "2020-01-03", "2020-02-03", "2020-02-04"],
        )
        metadata = json.loads((provider / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["future_calendar_end_date"], "20200210")
        start, close = _read_bin(provider / "features" / "sz000001" / "close.day.bin")
        self.assertEqual(start, 0)
        np.testing.assert_allclose(close, [1.0, 1.2, 1.4], rtol=1e-6)
        _, industry = _read_bin(provider / "features" / "sz000001" / "industry.day.bin")
        np.testing.assert_allclose(industry, [801010.0, 801010.0, 801020.0])
        self.assertFalse((provider / "features" / "sz000001" / "amount.day.bin").exists())
        self.assertFalse((provider / "features" / "sz000001" / "change.day.bin").exists())
        for index in INDICES:
            text = (provider / "instruments" / f"{index.market}.txt").read_text(encoding="utf-8")
            self.assertIn("SZ000001", text)

        refreshed = standard.copy()
        refreshed["turnover"] = [4.0, 5.0, 6.0]
        atomic_parquet(
            refreshed,
            self.config.standard_dir / "stocks" / "SZ000001.parquet",
        )
        refresh_provider_symbols({"SZ000001"}, self.config)
        _, turnover = _read_bin(provider / "features" / "sz000001" / "turnover.day.bin")
        np.testing.assert_allclose(turnover, [4.0, 5.0, 6.0])
        verify_provider(provider, self.config)

        import qlib
        from qlib.constant import REG_CN
        from qlib.data import D

        qlib.init(
            provider_uri=str(provider),
            region=REG_CN,
            expression_cache=None,
            dataset_cache=None,
        )
        features = D.features(
            ["SZ000001"],
            ["$close", "$factor", "$industry"],
            start_time="2020-01-02",
            end_time="2020-02-03",
            freq="day",
        )
        np.testing.assert_allclose(features["$close"], [1.0, 1.2, 1.4], rtol=1e-6)
        np.testing.assert_allclose(features["$factor"], [0.1, 0.2, 0.2], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
