# -*- coding; utf-8 -*-
"""
Project: simpleBacktest
File: base_riskmanager.PY
Time: 16:17
Date: 2022/6/17
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""

from simpleBacktest.base.metabase_env import EnvBase

class RiskManagerBase(EnvBase):

    def __init__(self):
        self.env.risk_managers.update({self.__class__.__name__: self})

    def run(self):
        pass