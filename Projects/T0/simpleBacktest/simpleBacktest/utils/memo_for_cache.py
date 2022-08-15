# -*- coding; utf-8 -*-
"""
Project: simpleBacktest
File: memo_for_cache.PY
Time: 15:28
Date: 2022/6/20
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""

from functools import wraps

from simpleBacktest.environment import Environment


def memo(key):
    cache = Environment.cache

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if key not in cache:
                cache[key] = func(*args, **kwargs)

            return cache[key]

        return wrapper

    return decorate