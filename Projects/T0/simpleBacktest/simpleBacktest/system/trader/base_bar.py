# -*- coding; utf-8 -*-
"""
Project: simpleBacktest
File: base_bar.PY
Time: 16:47
Date: 2022/6/17
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""
from copy import copy
from typing import Generator

import arrow

from simpleBacktest.base.metabase_env import EnvBase
from simpleBacktest.utils.easy_func import get_day_ratio, format_date

class BarBase(EnvBase):

    def __init__(self, ticker: str, frequency: str):

        self.frequency = frequency
        self.ticker = ticker

        self._iter_data: Generator = None
        self.previous_tick: dict = None
        self.current_tick: dict = None
        self.next_tick: dict = None

    @property
    def cur_price(self) -> float:
        return self.close

    @property
    def execute_price(self) -> float:
        if self.env.execute_on_close_or_next_open == 'open':
            return self.next_tick['open']

        return self.close

    @property
    def date(self) -> str:
        return self.current_tick['date']

    @property
    def open(self) -> float:
        return self.current_tick['open']

    @property
    def high(self) -> float:
        return self.current_tick['high']

    @property
    def low(self) -> float:
        return self.current_tick['low']

    @property
    def last(self) -> float:
        return self.current_tick['last']

    @property
    def volume(self) -> float:
        return self.current_tick['volume']

    @property
    def ask_price1(self) -> float:
        return self.current_tick['ask_price1']

    @property
    def bid_price1(self) -> float:
        return self.current_tick['bid_price1']


    def is_suspended(self) -> bool:  # 判断明天是否停牌
        now = arrow.get(self.env.sys_date)
        tomorrow = arrow.get(self.next_tick['date'])

        if tomorrow <= now:
            return False

        return True

    def next(self):
        """更新行情"""

        if self.is_suspended():
            self.env.cur_suspended_tickers.append(self.ticker)
            self.env.suspended_tickers_record[self.ticker].append(
                self.env.sys_date)
        else:
            self.next_directly()

    def next_directly(self):
        """不判断，直接next到下一个数据"""
        self.previous_tick = self.current_tick
        self.current_tick = self.next_tick
        self.next_tick = next(self._iter_data)

    def initialize(self, buffer_day: int) -> bool:
        sys_date = self.env.sys_date
        start = arrow.get(self.env.fromdate).shift(
            days=-buffer_day).format('YYYY-MM-DD HH:mm:ss')
        end = format_date(self.env.todate)

        if buffer_day > 500:  # 为了构建pre_tick而load,若生成不了就删除
            self._delete_tick('Delete TICK for PRE_TICK')

            return False

        if self._iter_data is None:  # 加载数据并初始化
            self._update_iter_data(start, end)

            try:
                for i in range(3):
                    self.next_directly()
            except StopIteration:
                self._delete_tick('Delete TICK for ALL')

                return False

        while arrow.get(self.next_tick.get('date')) <= arrow.get(sys_date):
            try:
                self.next_directly()  # sys_date为fromdate前一个周期,所以重复load数据到next_tick为fromdate
            except StopIteration:
                self._delete_tick('Delete TICK for SOME')

                return False

        if arrow.get(self.previous_tick['date']) >= arrow.get(sys_date):
            buffer_day += 300  # 更新好后，若当前pre_tick数据不对，表明
            self._iter_data = None  # next_tick数据也不对，要重新load

            return self.initialize(buffer_day)

        return True

    def _update_iter_data(self, start: str, end: str):
        reader = self.env.readers[self.ticker]
        self._iter_data = reader.load(start, end, self.frequency)

    def _delete_tick(self, message: str):
        """删除数据,并记录"""
        del self.env.readers[self.ticker]
        self.env.logger.warning(
            f'Delete {self.ticker}_{self.frequency} for lack of {message}!!!!')
        self.env.cur_suspended_tickers.append(self.ticker)
        self.env.suspended_tickers_record[self.ticker].append(
            self.env.sys_date)

    def move_next_tick_to_cur_tick(self):
        """
        用于伪造最新的next bar，骗过系统产生最新日期的信号
        会导致回测结果不准确。
        """
        date_format = "YYYY-MM-DD HH:mm:ss"
        next_date = arrow.get(self.next_tick['date']).format(date_format)
        todate = arrow.get(self.env.todate).format(date_format)

        if todate == next_date:
            self.current_tick = copy(self.next_tick)
            self.next_tick['date'] = arrow.get(todate).shift(
                days=get_day_ratio(self.env.sys_frequency)).format(date_format)
        else:
            self.env.cur_suspended_tickers.append(self.ticker)