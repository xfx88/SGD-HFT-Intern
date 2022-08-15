# -*- coding; utf-8 -*-
"""
Project: simpleBacktest
File: market_maker.PY
Time: 17:19
Date: 2022/6/17
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""
import arrow

from simpleBacktest.constants import EVENT
from simpleBacktest.system.components.exceptions import (BacktestFinished,
                                                    BlowUpError)
from simpleBacktest.base.metabase_env import EnvBase
from simpleBacktest.system.trader.base_bar import BarBase
from simpleBacktest.system.trader.calendar import Calendar


class MarketMaker(EnvBase):
    calendar: Calendar = None

    @classmethod
    def update_market(cls):
        try:
            cls.env.cur_suspended_tickers.clear() # 动态保存当前未更新数据的tickers
            cls.calendar.update_calendar()
            cls._update_bar()
            cls._update_recorder()
            cls._check_blowup()
            cls.env.event_engine.put(EVENT.Market_updated)

        except (BacktestFinished, BlowUpError):
            cls._update_recorder(final=True)  # 最后回测结束用close更新账户信息
            raise BacktestFinished

    @classmethod
    def initialize(cls):
        cls.env.logger.critical(f"AIMER正在初始化")
        cls._initialize_calendar()
        cls._initialize_feeds()
        cls._initialize_cleaners()
        cls.env.logger.critical(f"{'='*15} AIMER初始化成功！ {'='*15}")

    @classmethod
    def _initialize_calendar(cls):
        cls.calendar = Calendar(cls.env.instrument)

    @classmethod
    def _initialize_feeds(cls):
        for value in list(cls.env.readers.values()):
            if value.ticker:  # 若以key命名的，不作为ticker初始化
                ohlc_bar = cls.get_bar(value.ticker, cls.env.sys_frequency)

                if ohlc_bar.initialize(buffer_day=7):
                    cls.env.tickers.append(value.ticker)
                    cls.env.feeds.update({value.ticker: ohlc_bar})

    @classmethod
    def _initialize_cleaners(cls):
        for ticker in list(cls.env.tickers):
            for cleaner in list(cls.env.cleaners.values()):
                buffer_day = cleaner.buffer_day
                cleaner.initialize_buffer_data(ticker, buffer_day)

    @classmethod
    def _update_recorder(cls, final=False):
        for recorder in cls.env.recorders.values():
            recorder.update(order_executed=final)

    @classmethod
    def _check_blowup(cls):
        if cls.env.recorder.balance.latest() <= 0:
            cls.env.logger.critical("The account is BLOW UP!")
            raise BlowUpError

    @classmethod
    def _update_bar(cls):

        for ticker in cls.env.tickers:
            iter_bar = cls.env.feeds[ticker]

            try:
                iter_bar.next()
            except StopIteration:
                todate = arrow.get(cls.env.todate).format(
                    "YYYY-MM-DD HH:mm:ss")

                if cls.env.sys_date == todate:
                    if cls.env.is_show_today_signals:
                        iter_bar.move_next_ohlc_to_cur_ohlc()
                    else:
                        raise BacktestFinished
                else:
                    cls.env.cur_suspended_tickers.append(ticker)
                    cls.env.suspended_tickers_record[ticker].append(
                        cls.env.sys_date)

    @classmethod
    def get_bar(cls, ticker, frequency) -> BarBase:
        return cls.env.recorder.bar_class(ticker, frequency)