"""
提取 p2, p5, p18 至少有一个非0的样本
"""
import os
import shutil
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from functools import partial
import utilities as ut
from joblib import Parallel,delayed
import torch


loading_path = "/home/yby/SGD-HFT-Intern/Projects/T0/Data_labels/"
saving_path = "/home/yby/SGD-HFT-Intern/Projects/T0/Data_labels_sampled/"

month_list = ["07", "08", "09", "10", "11"]

factor_ret_cols = ['timeidx','price_pct','vwp_pct','spread','tick_spread','ref_ind_0','ref_ind_1','ask_weight_14',
                   'ask_weight_13','ask_weight_12','ask_weight_11','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
                   'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
                   'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
                   'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','bid_weight_11','bid_weight_12','bid_weight_13',
                   'bid_weight_14','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','turnover',
                   'tag', 'cls_2', 'cls_5', 'cls_18']

def get_file_list():
    dict_stock_dates = defaultdict(list)
    stock_list = os.listdir(loading_path)
    for s in stock_list:
        dict_stock_dates[s] = os.listdir(f"{loading_path}{s}/")
    return stock_list, dict_stock_dates

def train_dataset_rebuilder(stock, stock_date_list):
    # 测试数据库一般使用db=1
    rs = ut.redis_connection(db=1)
    concat_values = []
    current_month_id = 0
    stock_date_list.sort()

    for date in stock_date_list:
        df = pd.read_pickle(f"{loading_path}{stock}/{date}").reset_index()
        df["tag"] = df.apply(lambda x: 1 if sum(x[["cls_2", "cls_5", "cls_18"]]) else 0, axis = 1)
        data = df[factor_ret_cols]

        if date[4:6] == month_list[current_month_id]:
            concat_values.append(data)
        else:
            concat_values = pd.concat(concat_values)
            saving_file = f"{saving_path}{stock}/"
            if not os.path.exists(saving_file):
                os.makedirs(saving_file)
            ut.save_data_to_redis(rs, bytes(f'filterlabels_{stock}_{month_list[current_month_id]}', encoding = 'utf-8'), concat_values)

            concat_values = []
            current_month_id += 1
            if current_month_id == 4:
                break
            concat_values.append(data)

    rs.close()


def parallel_submit_data():
    stock_list, dick_stock_dates = get_file_list()
    Parallel(n_jobs=24, verbose=10, timeout=10000)(delayed(train_dataset_rebuilder)(stock, dick_stock_dates[stock])
                                                   for stock in stock_list)

if __name__ == "__main__":
    parallel_submit_data()