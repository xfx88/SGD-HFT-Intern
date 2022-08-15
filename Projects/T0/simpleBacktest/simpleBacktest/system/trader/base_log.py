# -*- coding; utf-8 -*-
"""
Project: simpleBacktest
File: base_log.PY
Time: 16:11
Date: 2022/6/20
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""
import abc

from dataclasses import dataclass, field

from simpleBacktest.constants import ActionType
from simpleBacktest.base.metabase_env import EnvBase
from simpleBacktest.system.trader.signals import SignalByTrigger


@dataclass
class TradeLogBase(EnvBase, abc.ABC):

    buy: float = None
    sell: float = None
    size: float = None

    entry_date: str = field(init=False)
    exit_date: str = field(init=False)

    entry_price: float = field(init=False)
    exit_price: float = field(init=False)

    entry_type: str = field(init=False)
    exit_type: str = field(init=False)

    pl_points: float = field(init=False)
    re_pnl: float = field(init=False)

    commission: float = field(init=False)

    @abc.abstractmethod
    def generate(self):
        raise NotImplementedError

    def _earn_short(self):
        return -1 if self.buy.action_type == ActionType.Short else 1

    @staticmethod
    def _get_order_type(order):
        if isinstance(order.signal, SignalByTrigger):
            return order.signal.order_type.value
        else:
            return order.order_type.value

    @abc.abstractmethod
    def settle_left_trade(self):
        raise NotImplementedError

    @property
    def ticker(self):
        return self.buy.ticker