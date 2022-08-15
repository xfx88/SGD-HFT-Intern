# -*- coding; utf-8 -*-
"""
Project: main.py
File: data_base.PY
Time: 9:36
Date: 2022/6/15
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""

import abc
from typing import Generator

from simpleBacktest.base.metabase_env import EnvBase


class ReaderBase(EnvBase, abc.ABC):

    def __init__(self, ticker: str, key: str = None) -> None:
        self.ticker = ticker
        self.key = key
        self.env.readers[ticker] = self

        if key:
            self.env.readers[f"{ticker}_{key}"] = self
        else:
            self.env.readers[ticker] = self

    @abc.abstractmethod
    def load(self, fromDate: str or int, endDate: str or int, frequency: str) -> Generator:
        raise NotImplementedError

    def load_by_cleaner(self, fromDate: str or int, endDate: str or int, frequency: str) -> Generator:

        return self.load(fromDate, endDate, frequency)