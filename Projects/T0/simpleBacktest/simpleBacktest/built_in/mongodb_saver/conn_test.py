# -*- coding; utf-8 -*-
"""
Project: simpleBacktest
File: conn_test.PY
Time: 13:47
Date: 2022/6/21
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""
from pymongo import MongoClient, mongo_client

if __name__ == "__main__":
    conn = MongoClient(host="192.168.1.139", port=27017, username="admin", password = "123456")
    pass