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

def gen_df_full_time():
    full_time = list(pd.date_range(start='09:30:00', end='11:30:00', freq='S'))
    full_time.extend(list(pd.date_range(start='13:00:00', end='15:00:00', freq='S')))
    full_time = [str(x.time()) for x in full_time]
    df_full_time = pd.DataFrame(index=full_time, columns={'price'})
    return df_full_time

def get_target(data, df_full_time):
    df_1s = df_full_time.copy()
    data = data.set_index('time')
    df_1s.price = data.vwp
    df_1s.price = df_1s.price.fillna(method='ffill')
    time_periods = ['1', '2', '5', '10', '20']
    for time_period in time_periods:
        df_1s[time_period] = df_1s.price.shift(-int(time_period) * 3) / df_1s.price - 1
    df_1s = df_1s.fillna(method = 'ffill')
    df_1s = df_1s.fillna(method = 'bfill')
    df_1s = df_1s.iloc[::3, -4:]

    return df_1s

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
    std_all = [[],[],[],[]]
    df_full_time = gen_df_full_time()

    maxlen = 4800
    values.sort()
    counter = 0
    value_list = []
    for v in values:
        # ??????????????????????????????
        df = pd.read_csv(f'{path}/{v}/{date}.csv')
        if len(df) < 10: continue
        df.columns = col_factors
        df = get_target(df, df_full_time).fillna(0)
        if len(v) > 4800: continue
        std = np.std(df.values, axis = 0)
        std_all[0].append(std[0])
        std_all[1].append(std[1])
        std_all[2].append(std[2])
        std_all[3].append(std[3])

    return std_all

def parallel_submit_date_numpy_train(db):
    """
    ??????????????????????????????????????????total_stock, total_tick, feature????????????numpy??????redis
    """
    date_ticker_dict = gen_date_ticker_dict(start_date=20210501, end_date=20210930)
    std_all = Parallel(n_jobs=24, verbose=2, timeout=10000)(delayed(submit_train_data)(date, values, db)
                                                                                for date, values in date_ticker_dict.items())
    std_2, std_5, std_10, std_20 = [], [], [], []
    for s in std_all:
        std_2.extend(s[0])
        std_5.extend(s[1])
        std_10.extend(s[2])
        std_20.extend(s[3])
    return


if __name__ == "__main__":
    parallel_submit_date_numpy_train(0)