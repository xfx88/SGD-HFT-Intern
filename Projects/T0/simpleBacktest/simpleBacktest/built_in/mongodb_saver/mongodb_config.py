# -*- coding; utf-8 -*-
"""
Project: simpleBacktest
File: mongodb_config.PY
Time: 10:53
Date: 2022/6/21
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""
import json
from datetime import datetime

import arrow
import funcy as fy
import pandas as pd
import pymongo

class MongodbConfig:

    host = "192.168.1.139"
    port = 27017
    username = "admin"
    password = "123456"
    dtformat = "%Y-%m-%d %H:%M:%S"
    tmformat = "%H:%M:%S"

    date                      = "date"
    code                      = "code"
    server_time               = "server_time"
    local_time                = "local_time"
    preclose                  = "preclose"
    open                      = "open"
    high                      = "high"
    low                       = "low"
    last                      = "last"
    upper_limit               = "upper_limit"
    lower_limit               = "lower_limit"
    volume                    = "volume"
    turnover                  = "turnover"
    iopv                      = "iopv"
    ask_price1,  bid_price1   = "ask_price1",  "bid_price1"
    ask_price2,  bid_price2   = "ask_price2",  "bid_price2"
    ask_price3,  bid_price3   = "ask_price3",  "bid_price3"
    ask_price4,  bid_price4   = "ask_price4",  "bid_price4"
    ask_price5,  bid_price5   = "ask_price5",  "bid_price5"
    ask_price6,  bid_price6   = "ask_price6",  "bid_price6"
    ask_price7,  bid_price7   = "ask_price7",  "bid_price7"
    ask_price8,  bid_price8   = "ask_price8",  "bid_price8"
    ask_price9,  bid_price9   = "ask_price9",  "bid_price9"
    ask_price10, bid_price10  = "ask_price10", "bid_price10"

    ask_volume1, bid_volume1   = "ask_volume1",  "bid_volume1"
    ask_volume2, bid_volume2   = "ask_volume2",  "bid_volume2"
    ask_volume3, bid_volume3   = "ask_volume3",  "bid_volume3"
    ask_volume4, bid_volume4   = "ask_volume4",  "bid_volume4"
    ask_volume5, bid_volume5   = "ask_volume5",  "bid_volume5"
    ask_volume6, bid_volume6   = "ask_volume6",  "bid_volume6"
    ask_volume7, bid_volume7   = "ask_volume7",  "bid_volume7"
    ask_volume8, bid_volume8   = "ask_volume8",  "bid_volume8"
    ask_volume9, bid_volume9   = "ask_volume9",  "bid_volume9"
    ask_volume10, bid_volume10 = "ask_volume10", "bid_volume10"

    client = pymongo.MongoClient(host = host, port = port, connect=False, username = username, password = password)

    def __init__(self, database, collection, host=None, port=None):
        self.host = host if host else self.host
        self.port = port if port else self.port
        self.database = database
        self.collection = collection

    def __set_dtformat(self, bar):
        """ 识别日期 """

        return arrow.get(bar["server_time"]).format("YYYY/MM/DD HH:mm:ss")

    def _set_collection(self):
        """设置数据库"""
        db = self.client[self.database]
        Collection = db[self.collection]

        return Collection

    def __load_csv(self, path):
        """读取CSV"""
        df = pd.read_csv(path)
        j = df.to_json()
        data = json.loads(j)

        return data

    def _combine_and_insert(self, data):
        """整合并插入数据"""
        # 构造 index 列表
        name_list = [
            self.date, self.code, self.server_time, self.local_time,
            self.preclose, self.open, self.high, self.low, self.last,
            self.upper_limit, self.lower_limit, self.volume, self.turnover, self.iopv,
            self.ask_price1 ,  self.ask_volume1,
            self.ask_price2 ,  self.ask_volume2,
            self.ask_price3 ,  self.ask_volume3,
            self.ask_price4 ,  self.ask_volume4,
            self.ask_price5 ,  self.ask_volume5,
            self.ask_price6 ,  self.ask_volume6,
            self.ask_price7 ,  self.ask_volume7,
            self.ask_price8 ,  self.ask_volume8,
            self.ask_price9 ,  self.ask_volume9,
            self.ask_price10,  self.ask_volume10,
            self.bid_price1 ,  self.bid_volume1 ,
            self.bid_price2 ,  self.bid_volume2 ,
            self.bid_price3 ,  self.bid_volume3 ,
            self.bid_price4 ,  self.bid_volume4 ,
            self.bid_price5 ,  self.bid_volume5 ,
            self.bid_price6 ,  self.bid_volume6 ,
            self.bid_price7 ,  self.bid_volume7 ,
            self.bid_price8 ,  self.bid_volume8 ,
            self.bid_price9 ,  self.bid_volume9 ,
            self.bid_price10,  self.bid_volume10
        ]

        def process_data(n):
            # 返回单个数据的字典，key为index，若无index则返回 None
            single_data = {index.lower(): data[index].get(str(n))
                           for index in name_list}

            return single_data

        length = len(data[self.date])  # 总长度
        coll = self._set_collection()

        # 插入数据

        for i in range(length):
            bar = process_data(i)
            bar[self.server_time.lower()] = self.__set_dtformat(bar)

            coll.insert_one(bar)

    def data_to_db(self, path):
        """数据导入数据库"""
        data = self.__load_csv(path)
        self._combine_and_insert(data)
        # print(f'{self.database}, {self.collection}, Total inserted: {len(data)}')
