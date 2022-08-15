# -*- coding; utf-8 -*-
"""
Project: simpleBacktest
File: calendar.PY
Time: 17:24
Date: 2022/6/17
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""
import arrow

from simpleBacktest.system.components.exceptions import BacktestFinished
from simpleBacktest.base.metabase_env import EnvBase
from simpleBacktest.utils.easy_func import get_day_ratio


class Calendar(EnvBase):

    def __init__(self, instrument):
        if instrument == "A_shares":
            self.is_trading_time = self._is_A_shares_trading_time

    def _is_A_shares_trading_time(self, now: arrow.arrow.Arrow) -> bool:
        weekday = now.isoweekday()
        date = now.format('YYYY-MM-DD')

        if self.env.sys_frequency == 'D':

            if weekday <= 5:
                return True

        else:

            if weekday <= 5:
                left_1 = arrow.get(f'{date} 09:30')
                right_1 = arrow.get(f'{date} 11:30')
                left_2 = arrow.get(f'{date} 13:00')
                right_2 = arrow.get(f'{date} 15:00')

                if left_1 <= now <= right_1 or left_2 <= now <= right_2:
                    return True

        return False

    def update_calendar(self):
        if self.env.is_live_trading:
            # 赋值system_date变量
            self.env.system_date = arrow.utcnow().format('YYYY-MM-DD HH:mm:ss')
        else:
            self._check_todate()
            ratio = get_day_ratio(self.env.system _frequency)
            new_sys_date = arrow.get(self.env.sys_date).shift(days=ratio)
            self.env.sys_date = new_sys_date.format('YYYY-MM-DD HH:mm:ss')

            while not self.is_trading_time(new_sys_date):
                self._check_todate()
                new_sys_date = arrow.get(self.env.sys_date).shift(days=ratio)
                self.env.sys_date = new_sys_date.format('YYYY-MM-DD HH:mm:ss')

    def _check_todate(self):
        if arrow.get(self.env.sys_date) >= arrow.get(self.env.todate):
            # TODO: 还有至少一个ticker时间超过
            raise BacktestFinished