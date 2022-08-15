# -*- coding; utf-8 -*-
"""
Project: simpleBacktest
File: Backtest.PY
Time: 9:48
Date: 2022/6/22
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""
import logging
import sys

import arrow

from simpleBacktest.built_in.optimizer import Optimizer
from simpleBacktest.config import EVENT_LOOP
from simpleBacktest.constants import EVENT
from simpleBacktest.custom.forward_analysis import ForwardAnalysis
from simpleBacktest.system.components.exceptions import BacktestFinished
from simpleBacktest.system.components.logger import LoggerFactory
from simpleBacktest.system.components.market_maker import MarketMaker
from simpleBacktest.system.components.order_checker import PendingOrderChecker
from simpleBacktest.system.components.output import OutPut
from simpleBacktest.base.metabase_env import EnvBase
from simpleBacktest.utils.awesome_func import show_process

class AIMER(EnvBase):

    def __init__(self):
        self.market_maker: MarketMaker = None
        self.pending_order_checker: PendingOrderChecker = None
        self.event_loop: list = None

        self.optimizer = Optimizer()
        self.forward_analysis = ForwardAnalysis()

    def _pre_initialize_trading_system(self):
        self.event_loop = EVENT_LOOP
        self.market_maker = MarketMaker()
        self.pending_order_checker = PendingOrderChecker()

    def initialize_trading_system(self):
        self._pre_initialize_trading_system()
        self.env.initialize_env() # 刷新environment缓存，重置计数器
        self.market_maker.initialize() # 初始化calendar, feeds 以及 clearner; feeds实为喂数据
        self.env.recorder.initialize() #

    def sunny(self, summary: bool = True, show_process: bool = False):
        self.initialize_trading_system()

        while True:
            try:
                if self.env.envent_engine.is_empty():
                    self.market_maker.update_market()
                    self.pending_order_checker.run()
                else:
                    cur_event = self.env.envent_engine.get()
                    self._run_event_loop(cur_event)

            except BacktestFinished:
                if summary:
                    print("\n")
                    self.output.summary()

                break

    def _run_event_loop(self, cur_event):
        for element in self.event_loop:
            if self._event_is_executed(cur_event, **element):
                break

    def _event_is_executed(
            self, cur_event, if_event: EVENT, then_event: EVENT, module_dict: dict) -> bool:
        if cur_event is None:
            return True

        elif cur_event == if_event:
            [value.run() for value in module_dict.values()]
            self.env.event_engine.put(then_event)

            return True

        else:
            return False

    def _show_process(self):
        fromDate = arrow.get(self.env.fromDate)
        endDate = arrow.get(self.env.endDate)
        curDate = arrow.get(self.env.system_date)
        total_days = (endDate - fromDate).days
        finished_days = (curDate - fromDate).days
        show_process(finished_days, total_days)

    def set_date(
            self, fromDate: str, endDate: str, frequency: str, instrument: str):

        self.env.instrument = instrument
        self.env.fromDate = fromDate
        self.env.endDate = endDate
        self.env.frequency = frequency

    @classmethod
    def show_log(cls, file = False, no_console = False):
        if file:
            LoggerFactory("AIMER")

        if no_console:
            logging.getLogger("AIMER").propagate = False

        logging.basicConfig(level = logging.INFO)

    @classmethod
    def set_recursion_limit(cls, limit: int = 2000):
        sys.setrecursionlimit(limit)

    def save_original_signal(self):
        self.env.is_save_original = True

    @property
    def output(self) -> OutPut:
        return OutPut()