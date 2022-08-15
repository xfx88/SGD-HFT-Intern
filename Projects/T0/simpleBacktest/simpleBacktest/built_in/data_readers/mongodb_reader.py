# -*- coding; utf-8 -*-
"""
Project: simpleBacktest
File: MongodbReader.PY
Time: 15:57
Date: 2022/6/22
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""

import time

import pymongo

from simpleBacktest.base.base_reader import ReaderBase


class MongodbReader(ReaderBase):

    host = "192.168.1.139"
    port = 27017
    username = "admin"
    password = "123456"
    client = pymongo.MongoClient(host, port, connect = False, username = username, password = password)

    def __init__(self, database, ticker, key = None):

        super().__init__(ticker, key)
        self.database = database

    def set_collection(self, database: str, collection: str):
        db = self.client[database]
        coll = db[collection]

        return coll

    def load(self, fromDate: str or int, endDate: str or int, frequency: str):
        if self.key:
            coll = self.set_collection(database = self.database,
                                       collection = self.key)
        else:
            coll = self.set_collection(database = self.database,
                                       collection = frequency)

        result = coll.find({"date":{"$gte": fromDate, "$lte": endDate}}).sort("date", 1)

        return result