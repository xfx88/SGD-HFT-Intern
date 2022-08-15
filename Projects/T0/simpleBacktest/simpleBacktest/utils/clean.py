# -*- coding; utf-8 -*-
"""
Project: simpleBacktest
File: clean.PY
Time: 15:55
Date: 2022/6/20
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""

from functools import wraps
import arrow

def make_it_float(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return float(func(*args, **kwargs))

    return wrapper


def make_it_datetime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return arrow.get(func(*args, **kwargs))

    return