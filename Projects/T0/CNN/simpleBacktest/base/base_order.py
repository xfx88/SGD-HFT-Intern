# -*- coding; utf-8 -*-
"""
Project: main.py
File: base_order.PY
Time: 16:04
Date: 2022/6/15
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""
import abc
from itertools import count
from typing import Union

from simpleBacktest.constants import ActionType, OrderStatus, OrderType
from simpleBacktest.base.metabase_env import EnvBase
from simpleBacktest.system.models.signals import Signal, SignalByTrigger, SignalCancelBase

class OrderBase(EnvBase, abc.ABC):

    _action_type: ActionType = None
    order_type: OrderType = None

    counter = count(1)

    def __init__(self, signal: Signal, mkt_id: int):
        self.signal: Signal = signal
        self.strategy_name = signal.strategy_name
        self.ticker:str = signal.ticker
        self.size: int = signal.size

        self.order_id: int = next(self.counter)
        self.mkt_id: int = mkt_id
        self.first_cur_price: float = self._get_first_cur_price()

        self.status = OrderStatus.Created
        self.trading_date = self.signal.datetime
        self._status: OrderStatus = None

    def _get_first_cur_price(self) -> float:
        if isinstance(self.signal, SignalByTrigger):
            return self.signal.execute_price

        return self.env.feeds[self.ticker].execute_price

    @abc.abstractmethod
    def action_type(self) -> ActionType:
        raise NotImplementedError

    @abc.abstractmethod
    def status(self) -> OrderStatus:
        raise NotImplementedError

    @status.setter
    def status(self, value: OrderStatus) -> None:
        self._status = value
        raise NotImplementedError

class PendingOrderBase(OrderBase):

    def __init__(self, signal: Signal, mkt_id: int, trigger_key: str) -> None:
        self.trigger_key = trigger_key
        super().__init__(signal, mkt_id)
        self.trading_date = self.env.feeds[signal.ticker].next["ohlc"]

    @property
    def action_type(self) -> ActionType:
        return self._action_type

    @property
    def status(self) -> OrderStatus:
        return self._status

    @status.setter
    def status(self, value: OrderStatus):