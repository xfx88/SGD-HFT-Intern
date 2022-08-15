# -*- coding; utf-8 -*-
"""
Project: simpleBacktest
File: backtest.PY
Time: 17:10
Date: 2022/6/22
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""

import simpleBacktest as bt
from simpleBacktest.built_in.backtest_stock.stock_broker import StockBroker
from simpleBacktest.built_in.backtest_stock.stock_recorder import StockRecorder

def stock(ticker_list: list, frequency: str,
          initial_cash: float, start: str, end: str, broker: str = "admin"):

    for ticker in ticker_list:
        bt.data_readers.MongodbReader(database = f"{broker}", ticker = ticker)

    StockBroker()

    StockRecorder().set_setting(initial_cash = initial_cash,
                                comm = None,
                                comm_pct = 0.0016,
                                margin_rate = 0.1)

    go = bt.AIMER()
    go.set_date(start, end, frequency, instrument = "AShares")

    return go