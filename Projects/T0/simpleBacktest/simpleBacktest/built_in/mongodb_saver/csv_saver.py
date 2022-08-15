# -*- coding; utf-8 -*-
"""
Project: simpleBacktest
File: csv_saver.PY
Time: 10:52
Date: 2022/6/21
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""
from simpleBacktest.built_in.mongodb_saver.mongodb_config import MongodbConfig
import os
from tqdm import tqdm
from joblib import Parallel, delayed


tick_data_path = "/home/wuzhihan/Projects/tickdata_temp"

class Tick_csv_to_MongoDB(MongodbConfig):
    host = "192.168.1.139"
    port = 27017
    username = "admin"
    password = "123456"

    dtformat = "%Y-%m-%d %H:%M:%S"
    tmformat = "%H:%M:%S"
    date = "date"
    code = "code"
    server_time = "server_time"
    local_time = "local_time"
    preclose = "preclose"
    open = "open"
    high = "high"
    low = "low"
    last = "last"
    upper_limit = "upper_limit"
    lower_limit = "lower_limit"
    volume = "volume"
    turnover = "turnover"
    iopv = "iopv"

    ask_price1, bid_price1 = "ask_price1", "bid_price1"
    ask_price2, bid_price2 = "ask_price2", "bid_price2"
    ask_price3, bid_price3 = "ask_price3", "bid_price3"
    ask_price4, bid_price4 = "ask_price4", "bid_price4"
    ask_price5, bid_price5 = "ask_price5", "bid_price5"
    ask_price6, bid_price6 = "ask_price6", "bid_price6"
    ask_price7, bid_price7 = "ask_price7", "bid_price7"
    ask_price8, bid_price8 = "ask_price8", "bid_price8"
    ask_price9, bid_price9 = "ask_price9", "bid_price9"
    ask_price10, bid_price10 = "ask_price10", "bid_price10"

    ask_volume1, bid_volume1 = "ask_volume1", "bid_volume1"
    ask_volume2, bid_volume2 = "ask_volume2", "bid_volume2"
    ask_volume3, bid_volume3 = "ask_volume3", "bid_volume3"
    ask_volume4, bid_volume4 = "ask_volume4", "bid_volume4"
    ask_volume5, bid_volume5 = "ask_volume5", "bid_volume5"
    ask_volume6, bid_volume6 = "ask_volume6", "bid_volume6"
    ask_volume7, bid_volume7 = "ask_volume7", "bid_volume7"
    ask_volume8, bid_volume8 = "ask_volume8", "bid_volume8"
    ask_volume9, bid_volume9 = "ask_volume9", "bid_volume9"
    ask_volume10, bid_volume10 = "ask_volume10", "bid_volume10"
    
    def __init__(self, database, collection, host=None, port=None):
        super(Tick_csv_to_MongoDB, self).__init__(database, collection, host, port)

def write_data_to_mongodb():
    db_writer = Tick_csv_to_MongoDB(database="admin", collection="test")

    ticker_list = os.listdir(tick_data_path)
    ticker_list.sort()

    def sub_writer(ticker):
        file_path = f"{tick_data_path}/{ticker}/"
        file_list = os.listdir(file_path)
        file_list = [f for f in file_list if f[11:13] == "11"]
        for f in file_list:
            db_writer.data_to_db(file_path + f)

    Parallel(n_jobs=40, verbose=0)(delayed(sub_writer)(ticker)
                                   for ticker in tqdm(ticker_list))

    return


if __name__ == "__main__":
    write_data_to_mongodb()