# -*- coding; utf-8 -*-
"""
Project: simpleBacktest
File: event_engine.PY
Time: 14:47
Date: 2022/6/17
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""
import queue

from simpleBacktest.constants import EVENT

class EventEngine:

    def __init__(self):
        self._core = queue.Queue()

    def put(self, event: EVENT):
        self._core.put(event)

    def get(self) -> EVENT:
        return self._core.get(block=False)

    def is_empty(self) -> bool:
        return self._core.empty()