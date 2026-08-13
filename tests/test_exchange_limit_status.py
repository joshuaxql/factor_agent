from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from qlib.backtest.decision import Order, OrderDir
from qlib.backtest.exchange import Exchange
from qlib.backtest.high_performance_ds import NumpyQuote
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.log import get_module_logger
from qlib.utils.index_data import SingleData


class LimitStatusExchangeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dates = pd.date_range("2020-01-01", periods=9, freq="D")
        index = pd.MultiIndex.from_product(
            [["TEST"], cls.dates], names=["instrument", "datetime"]
        )
        cls.exchange = object.__new__(Exchange)
        cls.exchange.buy_price = "$close"
        cls.exchange.sell_price = "$close"
        cls.exchange.limit_status_enabled = True
        cls.exchange.quote_df = pd.DataFrame(
            {
                "$open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                "$close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5],
                "$factor": [1.0] * 9,
                "$limit_status": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan, -1.0],
                "$volume": [1000.0] * 9,
            },
            index=index,
        )
        cls.exchange._update_limit_status()
        cls.exchange.quote = NumpyQuote(cls.exchange.quote_df, "day")
        cls.exchange.buy_vol_limit = None
        cls.exchange.sell_vol_limit = None
        cls.exchange.trade_unit = None
        cls.exchange.trade_w_adj_price = False
        cls.exchange.open_cost = 0.0
        cls.exchange.close_cost = 0.0
        cls.exchange.min_cost = 0.0
        cls.exchange.impact_cost = 0.0
        cls.exchange.logger = get_module_logger("limit_status_test")

    def _tradable(self, day: int, direction: OrderDir) -> bool:
        return self.exchange.is_stock_tradable(
            "TEST", self.dates[day], self.dates[day], direction=direction
        )

    def _price(self, day: int, direction: OrderDir) -> float:
        return self.exchange.get_deal_price(
            "TEST", self.dates[day], self.dates[day], direction=direction
        )

    def test_directional_tradability(self) -> None:
        self.assertTrue(self._tradable(2, OrderDir.BUY))
        self.assertTrue(self._tradable(2, OrderDir.SELL))
        self.assertFalse(self._tradable(3, OrderDir.BUY))
        self.assertTrue(self._tradable(3, OrderDir.SELL))
        self.assertTrue(self._tradable(5, OrderDir.BUY))
        self.assertTrue(self._tradable(5, OrderDir.SELL))
        self.assertTrue(self._tradable(6, OrderDir.BUY))
        self.assertFalse(self._tradable(6, OrderDir.SELL))
        self.assertFalse(self._tradable(7, OrderDir.BUY))
        self.assertFalse(self._tradable(7, OrderDir.SELL))
        self.assertFalse(self._tradable(8, OrderDir.BUY))
        self.assertFalse(self._tradable(8, OrderDir.SELL))

    def test_directional_open_price(self) -> None:
        self.assertEqual(self._price(2, OrderDir.BUY), 12.0)
        self.assertEqual(self._price(2, OrderDir.SELL), 12.5)
        self.assertEqual(self._price(5, OrderDir.BUY), 15.5)
        self.assertEqual(self._price(5, OrderDir.SELL), 15.0)
        self.assertEqual(self._price(1, OrderDir.BUY), 11.5)
        self.assertEqual(self._price(4, OrderDir.SELL), 14.5)

    def test_order_check_uses_direction(self) -> None:
        buy = Order("TEST", 100, Order.BUY, self.dates[3], self.dates[3])
        sell = Order("TEST", 100, Order.SELL, self.dates[3], self.dates[3])
        self.assertFalse(self.exchange.check_order(buy))
        self.assertTrue(self.exchange.check_order(sell))

        buy = Order("TEST", 100, Order.BUY, self.dates[6], self.dates[6])
        sell = Order("TEST", 100, Order.SELL, self.dates[6], self.dates[6])
        self.assertTrue(self.exchange.check_order(buy))
        self.assertFalse(self.exchange.check_order(sell))

    def test_deal_order_uses_open_and_blocks_one_word_limits(self) -> None:
        buy = Order("TEST", 100, Order.BUY, self.dates[2], self.dates[2])
        trade_val, trade_cost, trade_price = self.exchange.deal_order(buy)
        self.assertEqual(trade_price, 12.0)
        self.assertEqual(trade_val, 1200.0)
        self.assertEqual(trade_cost, 0.0)

        sell = Order("TEST", 100, Order.SELL, self.dates[5], self.dates[5])
        trade_val, trade_cost, trade_price = self.exchange.deal_order(sell)
        self.assertEqual(trade_price, 15.0)
        self.assertEqual(trade_val, 1500.0)
        self.assertEqual(trade_cost, 0.0)

        blocked_buy = Order("TEST", 100, Order.BUY, self.dates[3], self.dates[3])
        trade_val, trade_cost, trade_price = self.exchange.deal_order(blocked_buy)
        self.assertEqual((trade_val, trade_cost), (0.0, 0.0))
        self.assertTrue(np.isnan(trade_price))
        self.assertEqual(blocked_buy.deal_amount, 0.0)

        blocked_sell = Order("TEST", 100, Order.SELL, self.dates[6], self.dates[6])
        trade_val, trade_cost, trade_price = self.exchange.deal_order(blocked_sell)
        self.assertEqual((trade_val, trade_cost), (0.0, 0.0))
        self.assertTrue(np.isnan(trade_price))
        self.assertEqual(blocked_sell.deal_amount, 0.0)

    def test_time_series_deal_price_switches_per_day(self) -> None:
        buy = self.exchange.get_deal_price(
            "TEST", self.dates[0], self.dates[6], OrderDir.BUY, method=None
        )
        sell = self.exchange.get_deal_price(
            "TEST", self.dates[0], self.dates[6], OrderDir.SELL, method=None
        )
        self.assertIsInstance(buy, SingleData)
        self.assertIsInstance(sell, SingleData)
        np.testing.assert_allclose(buy.data, [10.5, 11.5, 12.0, 13.5, 14.5, 15.5, 16.5])
        np.testing.assert_allclose(sell.data, [10.5, 11.5, 12.5, 13.5, 14.5, 15.0, 16.5])

    def test_required_fields_exclude_change(self) -> None:
        self.assertNotIn("$change", self.exchange.quote_df.columns)
        self.assertIn("$limit_status", self.exchange.quote_df.columns)
        self.assertIn("$open", self.exchange.quote_df.columns)

    def test_disabled_limit_status_ignores_blocks_and_forced_open(self) -> None:
        self.exchange.limit_status_enabled = False
        try:
            for day in (2, 3, 5, 6, 7, 8):
                self.assertTrue(self._tradable(day, OrderDir.BUY))
                self.assertTrue(self._tradable(day, OrderDir.SELL))
            self.assertEqual(self._price(2, OrderDir.BUY), 12.5)
            self.assertEqual(self._price(5, OrderDir.SELL), 15.5)

            buy = Order("TEST", 100, Order.BUY, self.dates[3], self.dates[3])
            trade_val, trade_cost, trade_price = self.exchange.deal_order(buy)
            self.assertEqual((trade_val, trade_cost, trade_price), (1350.0, 0.0, 13.5))
        finally:
            self.exchange.limit_status_enabled = True

    def test_strategy_propagates_limit_status_setting(self) -> None:
        strategy = TopkDropoutStrategy(
            signal=pd.Series(dtype=float),
            topk=1,
            n_drop=1,
            limit_status_enabled=False,
        )

        strategy.reset_common_infra({"trade_exchange": self.exchange})

        self.assertFalse(self.exchange.limit_status_enabled)
        self.exchange.limit_status_enabled = True

        TopkDropoutStrategy(
            signal=pd.Series(dtype=float),
            topk=1,
            n_drop=1,
            limit_status_enabled=False,
            trade_exchange=self.exchange,
        )
        self.assertFalse(self.exchange.limit_status_enabled)
        self.exchange.limit_status_enabled = True


if __name__ == "__main__":
    unittest.main()
