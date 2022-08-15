# -*- coding; utf-8 -*-
"""
Project: simpleBacktest
File: optimizer.PY
Time: 10:45
Date: 2022/6/22
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""
import multiprocessing
import os
import time
from collections import defaultdict
from itertools import count, product
from typing import Iterable, Tuple

import pandas as pd

import simpleBacktest as bt
from simpleBacktest.base.metabase_env import EnvBase
from simpleBacktest.utils.awesome_func import run_multiprocessing


class Optimizer(EnvBase):

    def __init__(self):
        self.workers = os.cpu_count()
        self.initial_params = defaultdict(dict)
        self.mid_params = defaultdict(list)
        self.final_params = None
        self.strategy_names = []
        self.total_iter_times = None

    def refresh(self):
        self.initial_params = defaultdict(dict)
        self.mid_params = defaultdict(list)
        self.final_params = None
        self.strategy_names = []
        self.total_iter_times = None

    def _tuple_to_dict(self, tuple_list: Tuple[dict]):
        value = {}

        for i in tuple_list:
            value.update(i)

        return value

    def _optimize_func(self, params: dict, cache: list, index: int):
        timer_start = time.time()

        bt.CleanerBase.counter = count(1)

        go = bt.Aimer()

        for strategy_name, strategy in go.env.strategies.items():
            strategy.set_params(params[strategy_name])

        go.sunny(False)

        summary = go.output.analysis.general_summary()
        summary.update(params)
        cache.append(summary)

        timer_end = time.time()

        self._compute_running_time(timer_start, timer_end, len(cache))

    def _compute_running_time(self, start: float, end: float, finished_times: int):

        diff = end - start
        left = diff * (self.total_iter_times - finished_times) / 60 / self.workers

    def _combine_all_params(self):
        for name in self.strategy_names:
            strategy_params = product(*self.initial_params[name].values())

            for i in strategy_params:
                self.mid_params[name].append({name: self._tuple_to_dict(i)})

        result = product(*self.mid_params.values())
        result = [self._tuple_to_dict(i) for i in result]
        unique = []

        for i in range(len(result)):
            new = result.pop()
            if new not in unique:
                unique.append(new)

        self.final_params = unique

    def set_params(self, strategy_name: str, params: str, params_range: Iterable):
        if strategy_name not in self.strategy_names:
            self.strategy_names.append(strategy_name)

        self.initial_params[strategy_name][params] = [{param: i} for i in params_range]

    def run(self, filename: str = "optimize_result.pkl"):
        self._combine_all_params()
        self.total_iter_times = len(self.final_params)

        print(f"一共优化{self.total_iter_times}次")
        cache_list: list = multiprocessing.Manager().list()
        params = [(param, cache_list, index) for index, param in enumerate(self.final_params)]
        run_multiprocessing(self._optimize_func, params, self.workers)
        print("参数优化完成！")

        if filename:
            pd.to_pickle([i for i in cache_list], filename)

        return [i for i in cache_list]