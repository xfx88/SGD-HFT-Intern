# -*- coding; utf-8 -*-
"""
Project: simpleBacktest
File: config.PY
Time: 14:51
Date: 2022/6/17
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""

from simpleBacktest.constants import EVENT
from simpleBacktest.base.metabase_env import EnvBase

# 控制事件发生顺序
EVENT_LOOP = [dict(if_event=EVENT.Market_updated,
                   then_event=EVENT.Data_cleaned,
                   module_dict=EnvBase.env.cleaners),

              dict(if_event=EVENT.Data_cleaned,
                   then_event=EVENT.Signal_generated,
                   module_dict=EnvBase.env.strategies),

              dict(if_event=EVENT.Signal_generated,
                   then_event=EVENT.Submit_order,
                   module_dict=EnvBase.env.risk_managers),

              dict(if_event=EVENT.Submit_order,
                   then_event=EVENT.Record_result,
                   module_dict=EnvBase.env.brokers),

              dict(if_event=EVENT.Record_result,
                   then_event=None,
                   module_dict=EnvBase.env.recorders)]