# !pip install paramiko
import os
from collections import defaultdict

import paramiko
import pandas as pd
# import modin.pandas as pd


class RemoteSrc:

    # 147上的路径
    REMOTE_PATH = "/sgd-data/data/stock/"
    # 缓存路径，方便复用，减少网络通讯
    TEMP = "/home/yby/arbitrage/backtest_temp/"

    def __init__(self):
        self._client = paramiko.Transport(("192.168.1.147", 22))
        self._client.connect(username="sgd", password="sgd123")
        # 使用sftp文件服务
        self._SFTP = paramiko.SFTPClient.from_transport(self._client)

        if not os.path.exists(self.TEMP):
            os.mkdir(self.TEMP)

        self.dict_stocksPerDay = defaultdict(list)

    def get_raw_bars(self, ticker, date):

        # 本地文件名，用于判断此前是否查询调用过
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

        # 数据字段处理，按自己需要修改
        data['server_time'] = pd.to_datetime(data.server_time)
        # data['local_time'] = data['server_time']
        data['time'] = data.apply(lambda x: str(x['server_time'].time()), axis = 1)

        return data

    def get_stock_list(self, date):
        files_currentDay = self._SFTP.listdir(f"{self.REMOTE_PATH}{date}/tick_csv/")
        return files_currentDay
        # if date in self.dict_stocksPerDay.keys():
        #     stocks_currentDay = self.dict_stocksPerDay[date]
        # else:
        #     stocks_currentDay = [s[:6] for s in files_currentDay]
        # return stocks_currentDay