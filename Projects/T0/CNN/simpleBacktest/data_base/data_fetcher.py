# -*- coding; utf-8 -*-
"""
Project: main.py
File: data_fetcher.PY
Time: 16:50
Date: 2022/6/14
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""


import os
import paramiko
from collections import defaultdict
import pandas as pd

class RemoteSrc:
    dict_path = "./"
    REMOTE_PATH = "/sgd-data/data/stock/"
    TEMP = "/home/yby/YBY/CNN/backtest_temp/"

    def __init__(self):
        self._client = paramiko.Transport(("192.168.1.147", 22))
        self._client.connect(username="sgd", password="sgd123")
        self._SFTP = paramiko.SFTPClient.from_transport(self._client)
        if not os.path.exists(self.TEMP):
            os.mkdir(self.TEMP)

        self.dict_stocksPerDay = defaultdict(list)

    def get_raw_bars(self, ticker, date):

        local_path = f"{self.TEMP}{ticker}_{date}.csv.gz"

        if not os.path.exists(local_path):
            files_currentDay = self._SFTP.listdir(f"{self.REMOTE_PATH}{date}/tick_csv/")
            if date in self.dict_stocksPerDay.keys():
                stocks_currentDay = self.dict_stocksPerDay[date]
            else:
                stocks_currentDay = [s[:6] for s in files_currentDay]

            file_idx = stocks_currentDay.index(ticker)

            self._SFTP.get(remotepath=f"{self.REMOTE_PATH}{date}/tick_csv/{files_currentDay[file_idx]}",
                           localpath=local_path)

        data = pd.read_csv(local_path)
        data['server_time'] = pd.to_datetime(data.server_time)
        data['local_time'] = data['server_time']
        data['time'] = data.apply(lambda x: str(x['server_time'].time()), axis = 1)

        return data