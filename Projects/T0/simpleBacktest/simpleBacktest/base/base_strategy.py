# -*- coding; utf-8 -*-
"""
Project: simpleBacktest
File: base_strategy.PY
Time: 14:36
Date: 2022/6/17
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""

import abc

from simpleBacktest.constants import ActionType
from simpleBacktest.system.base_recorder import RecorderBase
from simpleBacktest.system.components.signal_generator import SignalGenerator
from simpleBacktest.base.metabase_env import EnvBase

class StrategyBase(EnvBase, abc.ABC):

    def __init__(self) -> None:
        self.name = self.__class__.__name__
        self.env.strategies.update(({self.name: self}))

        self.buy = SignalGenerator(ActionType.Buy, self.name)
        self.sell = SignalGenerator(ActionType.Sell, self.name)
        self.buyclose = SignalGenerator(ActionType.BuyClose, self.name)
        self.sellclose = SignalGenerator(ActionType.SellClose, self.name)

        self.cancel_pending = SignalGenerator(ActionType.Cancel, self.name).cancel_pending
        self.params: dict = {}

    @property
    def recorder(self) -> RecorderBase:
        return  self.env.recorder

    @abc.abstractmethod
    def handle_bar(self):
        raise NotImplementedError

    def run(self):
        self.handle_bar()

    def cur_price(self, ticker: str) -> float:
        return self.env.feeds[ticker].cur_price

    def set_params(self, params: dict):
        self.params.update(params)