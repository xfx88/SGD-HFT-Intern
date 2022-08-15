# -*- coding; utf-8 -*-
"""
Project: main.py
File: __init__.py.PY
Time: 16:47
Date: 2022/6/14
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""
from simpleBacktest.built_in import data_readers
from simpleBacktest.built_in.backtest_stock import StockBroker
from simpleBacktest.Backtest import AIMER

from simpleBacktest.base.base_cleaner import CleanerBase
from simpleBacktest.base.base_broker import BrokerBase
from simpleBacktest.base.base_reader import ReaderBase
from simpleBacktest.base.base_recorder import RecorderBase
from simpleBacktest.base.base_riskmanager import RiskManagerBase
from simpleBacktest.base.base_strategy import StrategyBase
from simpleBacktest.system.trader.base_bar import BarBase
from simpleBacktest.system.trader.calendar import Calendar

__author__ = "Zhihan WU"
__version__ = "0.1"

