# !pip install paramiko
import os
from collections import defaultdict
import paramiko
import pandas as pd
# import modin.pandas as pd

class RemoteSrc:

    # 147上的路径
    REMOTE_PATH_l1 = "/sgd-data/data/future/"
    REMOTE_PATH_l2 = "/sgd-data/data/future_l2/"
    # 缓存路径，方便复用，减少网络通讯
    TEMP_l1 = "/home/yby/SGD-HFT-Intern/Projects/CTA/backtest_temp/l1/"
    TEMP_l2 = "/home/yby/SGD-HFT-Intern/Projects/CTA/backtest_temp/l2/"

    def __init__(self):
        self._client = paramiko.Transport(("192.168.1.147", 22))
        self._client.connect(username="sgd", password="sgd123")
        # 使用sftp文件服务
        self._SFTP = paramiko.SFTPClient.from_transport(self._client)

        self.REMOTE_PATH = None
        os.makedirs(self.TEMP_l1, exist_ok=True)
        os.makedirs(self.TEMP_l2, exist_ok=True)

        self.future_day_dict = defaultdict(list)

    def get_raw_bars(self, ticker, date, level):
        # 判断需要的是level1还是level2行情
        if level not in ['l1', 'l2']:
            raise ValueError("Either ``l1`` or ``l2`` can be specified for parameter ``level``.")

        # 本地文件名，用于判断此前是否查询调用过
        local_path = f"{self.TEMP_l1}{ticker}_{date}.csv.gz" if level == 'l1' else f"{self.TEMP_l2}{ticker}_{date}.csv.gz"
        self.REMOTE_PATH = self.REMOTE_PATH_l1 if level == 'l1' else self.REMOTE_PATH_l2

        # 本地文件名，用于判断此前是否查询调用过
        local_path = f"{self.TEMP_l1}{ticker}_{date}.csv.gz"
        if not os.path.exists(local_path):
            files_currentDay = self._SFTP.listdir(f"{self.REMOTE_PATH}{date}/tick_csv/")
            if date in self.future_day_dict.keys():
                future_current_day = self.future_day_dict[date]
            else:
                future_current_day = [s.split('.')[0] for s in files_currentDay]
            file_idx = future_current_day.index(ticker)

            self._SFTP.get(remotepath=f"{self.REMOTE_PATH}{date}/tick_csv/{files_currentDay[file_idx]}",
                        localpath=local_path)

        data = pd.read_csv(local_path)

        # 数据字段处理，按自己需要修改
        data['server_time'] = pd.to_datetime(data.server_time)
        # data['local_time'] = data['server_time']
        data['time'] = data.apply(lambda x: str(x['server_time'].time()), axis = 1)

        return data

    def get_future_list(self, date, level):
        # 判断需要的是level1还是level2行情
        if level not in ['l1', 'l2']:
            raise ValueError("Either ``l1`` or ``l2`` can be specified for parameter ``level``.")

        # 本地文件名，用于判断此前是否查询调用过
        self.REMOTE_PATH = self.REMOTE_PATH_l1 if level == 'l1' else self.REMOTE_PATH_l2
        files_currentDay = self._SFTP.listdir(f"{self.REMOTE_PATH}{date}/tick_csv/")
        return files_currentDay
        # if date in self.future_day_dict.keys():
        #     future_current_day = self.future_day_dict[date]
        # else:
        #     future_current_day = [s[:6] for s in files_currentDay]
        # return future_current_day