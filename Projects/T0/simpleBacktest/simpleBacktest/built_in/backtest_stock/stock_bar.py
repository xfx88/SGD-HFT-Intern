# -*- coding; utf-8 -*-
"""
Project: simpleBacktest
File: stock_bar.PY
Time: 15:37
Date: 2022/6/22
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""
from simpleBacktest.system.trader.base_bar import BarBase


class BarAShares(BarBase):

    @property
    def pre_date(self) -> str:
        return self.previous_tick["date"]

    @property
    def pre_open(self) -> bool:
        return self.previous_tick["open"]

    @property
    def pre_high(self) -> bool:
        return self.previous_tick["high"]

    @property
    def pre_low(self) -> bool:
        return self.previous_tick["low"]

    @property
    def pre_last(self) -> float:
        return self.current_tick['last']

    @property
    def pre_volume(self) -> float:
        return self.current_tick['volume']

    @property
    def ask_price1(self) -> float:
        return self.current_tick['ask_price1']

    @property
    def bid_price1(self) -> float:
        return self.current_tick['bid_price1']