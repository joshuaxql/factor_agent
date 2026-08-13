from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from qlib.backtest.decision import OrderDir
from qlib.contrib import evaluate


class FakeExchange:
    def __init__(self) -> None:
        self.profits = {"A": 0.1, "B": 0.2}
        self.tradability_checks = []
        self.suspension_checks = []

    def is_stock_tradable(self, stock_id, start_time, end_time, direction):
        self.tradability_checks.append((stock_id, direction))
        return True

    def check_stock_suspended(self, stock_id, start_time, end_time):
        self.suspension_checks.append(stock_id)
        return False

    def get_quote_info(self, stock_id, start_time, end_time, field):
        return self.profits[stock_id]


class LongShortBacktestTest(unittest.TestCase):
    def test_uses_instruments_for_market_return(self) -> None:
        date = pd.Timestamp("2020-01-02")
        index = pd.MultiIndex.from_product(
            [[date], ["A", "B"]], names=["datetime", "instrument"]
        )
        pred = pd.DataFrame({"score": [1.0, 0.0]}, index=index)
        exchange = FakeExchange()

        with (
            patch.object(evaluate, "get_exchange", return_value=exchange),
            patch.object(evaluate.D, "calendar", return_value=np.array([date])),
            patch.object(evaluate, "get_date_range", return_value=np.array([date])),
        ):
            result = evaluate.long_short_backtest(
                pred,
                topk=1,
                deal_price="close",
                shift=0,
                open_cost=0,
                close_cost=0,
                min_cost=0,
                extract_codes=True,
            )

        self.assertAlmostEqual(result["long"].iloc[0], -0.05)
        self.assertAlmostEqual(result["short"].iloc[0], -0.05)
        self.assertAlmostEqual(result["long_short"].iloc[0], -0.1)
        self.assertEqual(exchange.suspension_checks, ["A", "B"])
        self.assertEqual(
            exchange.tradability_checks,
            [("A", OrderDir.BUY), ("B", OrderDir.SELL)],
        )


if __name__ == "__main__":
    unittest.main()
