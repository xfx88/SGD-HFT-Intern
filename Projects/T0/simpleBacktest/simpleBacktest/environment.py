# -*- coding; utf-8 -*-
"""
Project: main.py
File: context.PY
Time: 9:50
Date: 2022/6/15
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""
import logging

from collections import defaultdict
from typing import List, Dict
import arrow

import simpleBacktest as sbt
from simpleBacktest.event_engine import EventEngine
from simpleBacktest.utils.easy_func import  get_day_ratio

class Environment:

    system_date: str = None
    system_frequency: str = "tick"
    instrument: str = "stock"
    fromDate: str = None
    toDate: str = None
    tickers: list = []

    cache: dict = None

    # general settings
    execute_on_close_or_next_open: str = "open"
    is_save_original: bool = False
    is_live_trading: bool = False
    is_show_today_signals: bool = False

    # 回测各模块的字典
    readers: dict = {}
    feeds: dict = {}
    cleaners: dict = {}
    cleaner_feeds: dict = {}
    strategies: dict = {}
    brokers: dict = {}
    risk_managers: dict = {}
    recorders: dict = {}
    recorder = None

    # system memory
    signals_normal: list = []
    signals_pending: list = []
    signals_trigger: list = []
    signals_cancel: list = []

    # temple_signals
    signals_normal_cur: list = []
    signals_pending_cur: list = []
    signals_trigger_cur: list = []
    signals_cancel_cur: list = []

    orders_mkt_normal_cur: list = []
    orders_child_of_mkt_dict: list = []
    orders_mkt_absolute_cur: list = []
    orders_mkt_submitted_cur: list = []

    orders_pending: list = [] # 动态保持挂单，触发会删除

    orders_cancel_cur: list = [] # 动态保持撤单，会刷新
    orders_cancel_submitted_cur : list = [] #动态保存撤单，会刷新

    cur_suspended_tickers: list = [] # 动态保存当前未更新数据的tickers
    suspended_tickers_record: defaultdict = defaultdict(list)

    #system
    logger = logging.getLogger("AIMER")
    event_engine = EventEngine()
    cache: dict = {}

    @classmethod
    def initialize_env(cls):
        """刷新environment防止缓存累积"""
        cls.signals_normal.clear()
        cls.signals_pending.clear()
        cls.signals_trigger.clear()
        cls.signals_cancel.clear()
        cls.signals_normal_cur.clear()
        cls.signals_pending_cur.clear()
        cls.signals_trigger_cur.clear()
        cls.signals_cancel_cur.clear()
        cls.orders_mkt_normal_cur.clear()
        cls.orders_mkt_absolute_cur.clear()
        cls.orders_mkt_submitted_cur.clear()
        cls.orders_pending.clear()
        cls.orders_child_of_mkt_dict.clear()
        cls.orders_cancel_cur.clear()
        cls.orders_cancel_submitted_cur.clear()
        cls.tickers.clear()
        cls.cur_suspended_tickers.clear()
        cls.suspended_tickers_record.clear()
        cls.cache.clear()

        if not cls.is_live_trading:
            ratio = get_day_ratio(cls.system_frequency)
            cls.sys_date = arrow.get(cls.fromDate).shift(
                days=-ratio).format('YYYY-MM-DD HH:mm:ss')
        cls.reset_all_counters()

    @classmethod
    def clear_modules(cls):
        """刷新environment防止缓存累积"""
        cls.sys_date: str = None
        cls.sys_frequency: str = None

        cls.instrument: str = None
        cls.fromDate: str = None
        cls.toDate: str = None
        cls.tickers: list = []
        cls.cur_suspended_tickers: list = []
        cls.suspended_tickers_record: defaultdict = defaultdict(list)

        cls.market_maker = None
        cls.readers: dict = {}
        cls.feeds: dict = {}
        cls.cleaners: dict = {}
        cls.cleaners_feeds: dict = {}
        cls.strategies: dict = {}
        cls.brokers: dict = {}
        cls.risk_managers: dict = {}
        cls.recorders: dict = {}
        cls.recorder = None  # type: sbt.RecorderBase

        cls.event_loop = None  # type:  List[Dict]
        cls.cache = {}

        cls.execute_on_close_or_next_open: str = 'open'
        cls.is_save_original: bool = False
        cls.is_live_trading: bool = False
        cls.is_show_today_signals: bool = False

    @classmethod
    def reset_all_counters(cls):
        from itertools import count
        from simpleBacktest.system.trader import signals
        from simpleBacktest.base.base_cleaner import CleanerBase
        from simpleBacktest.system.trader.orders.base_order import OrderBase
        from simpleBacktest.system.components.order_generator import OrderGenerator

        CleanerBase.counter = count(1)
        signals.Signal.counter = count(1)
        signals.SignalByTrigger.counter = count(1)
        signals.SignalForPending.counter = count(1)
        signals.SignalCancelTST.counter = count(1)
        signals.SignalCancelPending.counter = count(1)
        OrderBase.counter = count(1)
        OrderGenerator.counter = count(1)