import os
from collections import defaultdict

import numpy as np

from datetime import datetime

import tst.utilities as ut
from joblib import Parallel,delayed
import pandas as pd
import rqdatac as rq
rq.init(15626436420, 'vista2525')

col_factors = ['date', 'time', 'timeidx', 'price', 'vwp', 'ask_price', 'bid_price', 'ask_price2', 'bid_price2',
               'ask_price4', 'bid_price4', 'ask_price8', 'bid_price8', 'spread', 'tick_spread',
               'ref_ind_0', 'ref_ind_1', 'ask_weight_14', 'ask_weight_13', 'ask_weight_12', 'ask_weight_11',
               'ask_weight_10', 'ask_weight_9',
               'ask_weight_8', 'ask_weight_7', 'ask_weight_6', 'ask_weight_5', 'ask_weight_4',
               'ask_weight_3', 'ask_weight_2', 'ask_weight_1', 'ask_weight_0', 'bid_weight_0',
               'bid_weight_1', 'bid_weight_2', 'bid_weight_3', 'bid_weight_4', 'bid_weight_5',
               'bid_weight_6', 'bid_weight_7', 'bid_weight_8', 'bid_weight_9', 'bid_weight_10',
               'bid_weight_11', 'bid_weight_12', 'bid_weight_13', 'bid_weight_14', 'ask_dec', 'bid_dec',
               'ask_inc', 'bid_inc', 'ask_inc2', 'bid_inc2', 'preclose', 'limit', 'turnover']

factor_ret_cols = ['timeidx', 'price', 'vwp', 'spread', 'tick_spread', 'ref_ind_0', 'ref_ind_1',
                   'ask_weight_14', 'ask_weight_13', 'ask_weight_12', 'ask_weight_11',
                   'ask_weight_10', 'ask_weight_9', 'ask_weight_8', 'ask_weight_7',
                   'ask_weight_6', 'ask_weight_5', 'ask_weight_4', 'ask_weight_3',
                   'ask_weight_2', 'ask_weight_1', 'ask_weight_0', 'bid_weight_0',
                   'bid_weight_1', 'bid_weight_2', 'bid_weight_3', 'bid_weight_4',
                   'bid_weight_5', 'bid_weight_6', 'bid_weight_7', 'bid_weight_8',
                   'bid_weight_9', 'bid_weight_10', 'bid_weight_11', 'bid_weight_12',
                   'bid_weight_13', 'bid_weight_14', 'ask_dec', 'bid_dec', 'ask_inc',
                   'bid_inc', 'ask_inc2', 'bid_inc2', '1', '2', '5', '10', '20']


path = '/sgd-data/t0_data/500factor/500factors'


def gen_date_ticker_dict(start_date = 20210501, end_date = 20210930):
    trading_dates = rq.get_trading_dates(start_date=start_date, end_date = end_date)
    trading_dates = list(map(lambda x: datetime.strftime(x, "%Y%m%d"), trading_dates))

    date_ticker_dict = defaultdict(list)
    tickers = os.listdir(path)
    for ticker in tickers:
        date_list = os.listdir(f'{path}/{ticker}/')
        for date in date_list:
            date = date[:8]
            if date in trading_dates:
                date_ticker_dict[date].append(ticker)

    return date_ticker_dict


def gen_processed_data_to_redis(file_path, df_full_time, db):

    data = pd.read_csv(file_path)
    data.columns = col_factors
    code = file_path.split('/')[-2]
    date = file_path.split('/')[-1][:-4]
    data.insert(2, 'code', code)
    data = ut.get_target(data, df_full_time)[factor_ret_cols].fillna(0)
    data = data.values.astype(np.float32)
    if len(data) == 0 or len(data) > 4800:
        return
    rs = ut.redis_connection(db)
    ut.save_data_to_redis(rs, b'numpy_' + bytes(f'{date}', encoding = 'utf-8'), data)
    rs.close()
    return


def submit_train_data(date, values, db):
    """
    shape: [stock_num, joint_tick_num, features]
    """
    df_full_time = ut.gen_df_full_time()

    maxlen = 4800
    values.sort()
    counter = 0
    value_list = []
    for v in values:
        # 过滤掉无效长度的数据
        df = pd.read_csv(f'{path}/{v}/{date}.csv')
        if len(df) < 10: continue
        df.columns = col_factors
        df = ut.get_target(df, df_full_time).fillna(0)
        if len(v) > 4800: continue
        df = df[factor_ret_cols]
        value_list.append(df.values)

    values = []
    for v in value_list:
        if not len(v) >= 4800:
            v = np.pad(v, ((0, maxlen - len(v)), (0, 0)), mode='constant')
            v = v.reshape((1, -1, v.shape[-1]))
        values.append(v.astype(np.float32))
    try:
        concat_value = np.concatenate(values, axis = 0)
    except:
        print("here")
    rs = ut.redis_connection(db=db)
    ut.save_data_to_redis(rs, b'numpy' + b'_' + bytes(f'{date}', encoding = 'utf-8'), concat_value)
    rs.close()
    return

def parallel_submit_date_numpy_train(db):
    """
    用于按时间读取数据，转换成（total_stock, total_tick, feature）形状的numpy存入redis
    """
    date_ticker_dict = gen_date_ticker_dict()
    Parallel(n_jobs=36, verbose=2, timeout=10000)(delayed(submit_train_data)(date, values, db)
                                                  for date, values in date_ticker_dict.items())
    return


if __name__ == "__main__":
    parallel_submit_date_numpy_train(0)