# -*- coding; utf-8 -*-
"""
Project: simpleBacktest
File: exception.PY
Time: 16:28
Date: 2022/6/17
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""


class BlowUpError(Exception):
    pass


class BacktestFinished(Exception):
    pass


class OrderConflictError(Exception):
    pass


class PctRangeError(Exception):
    pass