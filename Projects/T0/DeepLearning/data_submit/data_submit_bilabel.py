import os
import shutil
from collections import defaultdict, deque

import numpy as np

from datetime import datetime
from functools import partial

import utilities as ut
from joblib import Parallel,delayed
import pandas as pd
import rqdatac as rq
rq.init(15626436420, 'vista2525')

col_factors = ['date','code','timeidx','price','vwp','ask_price','bid_price','ask_price2','bid_price2','ask_price4',
               'bid_price4','ask_price8','bid_price8','spread','tick_spread','ref_ind_0','ref_ind_1','ask_weight_14',
               'ask_weight_13','ask_weight_12','ask_weight_11','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
               'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
               'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
               'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','bid_weight_11','bid_weight_12','bid_weight_13',
               'bid_weight_14','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','preclose','limit','turnover',
               'circulation_mv', 'p_2','p_5','p_18','p_diff']

factor_ret_cols = ['timeidx','price_pct','vwp_pct','spread','tick_spread','ref_ind_0','ref_ind_1','ask_weight_14',
                   'ask_weight_13','ask_weight_12','ask_weight_11','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
                   'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
                   'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
                   'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','bid_weight_11','bid_weight_12','bid_weight_13',
                   'bid_weight_14','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','turnover',
                   'cls_2', 'cls_5', 'cls_18']

related_mv_cols = ['ask_weight_14', 'ask_weight_13', 'ask_weight_12', 'ask_weight_11',
                   'ask_weight_10', 'ask_weight_9', 'ask_weight_8', 'ask_weight_7',
                   'ask_weight_6', 'ask_weight_5', 'ask_weight_4', 'ask_weight_3',
                   'ask_weight_2', 'ask_weight_1', 'ask_weight_0', 'bid_weight_0',
                   'bid_weight_1', 'bid_weight_2', 'bid_weight_3', 'bid_weight_4',
                   'bid_weight_5', 'bid_weight_6', 'bid_weight_7', 'bid_weight_8',
                   'bid_weight_9', 'bid_weight_10', 'bid_weight_11', 'bid_weight_12',
                   'bid_weight_13', 'bid_weight_14', 'ask_dec', 'bid_dec', 'ask_inc',
                   'bid_inc', 'ask_inc2', 'bid_inc2','turnover']

def gen_mapping():
    cnt = 0
    classifier_mappping = {}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                classifier_mappping[(i, j, k)] = cnt
                cnt += 1
    return classifier_mappping

MAPPING = gen_mapping()

path = '/home/yby/Data/'
tgt_path = '/home/yby/Data/'
saving_path = '/home/yby/Data_labels/'

def move_files():
    files = os.listdir(path)
    for f in files:
        ticker = f.split("_")[1][:6]
        date = f.split("_")[0]
        current_path = path + "/" + f
        target_path = f"{tgt_path}{ticker}/"
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        shutil.move(current_path, f"{target_path}/{date}.pkl")


def gen_date_ticker_dict(start_date = 20210701, end_date = 20211031):
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

def rotate_key_value_monthly(kv_dict):

    vk_dict = defaultdict(partial(defaultdict, list))
    for k, v in kv_dict.items():
        month = k[4:6]
        for value in v:
            vk_dict[month][value].append(k)

    return vk_dict

def submit_train_data(month, ticker, values, db):
    """
    shape: [stock_num, joint_tick_num, features]
    """

    value_list = []
    rs = ut.redis_connection(db=db)

    values.sort()
    for date in values:
        # ??????????????????????????????
        df = pd.read_pickle(f'{path}{ticker}/{date}.pkl').reset_index()

        if len(df) < 10: continue
        if len(df) > 4800: continue

        df['date_time'] = df.apply(lambda x: str(x['date']) + ' ' + x['time'], axis = 1)
        df['date_time'] = pd.to_datetime(df['date_time'])

        df['cls_2'] = df.apply(lambda x: 1 if x['p_2'] > 0 else 0, axis = 1)
        df['cls_5'] = df.apply(lambda x: 1 if x['p_5'] > 0 else 0, axis = 1)
        df['cls_18'] = df.apply(lambda x: 1 if x['p_18'] > 0 else 0, axis = 1)

        # df.set_index('date_time', inplace=True)

        # df['cls_2'] = df['price'][::-1] \
        #                   .rolling(2, closed='both') \
        #                   .apply(lambda x: judger(x, 0.001))[::-1]
        #
        # df['cls_5'] = df['price'][::-1] \
        #                   .rolling(5, closed='both') \
        #                   .apply(lambda x: judger(x, 0.0015))[::-1]
        #
        # df['cls_18'] = df['price'][::-1] \
        #                    .rolling('54s', closed='both') \
        #                    .apply(lambda x: judger(x, 0.0025))[::-1]

        df[['price_pct', 'vwp_pct']] = (df[['price', 'vwp']].div(df.preclose, axis=0) - 1) * 1000
        df['timeidx'] = (df.timeidx - 7114) / 4100  # normalize timeidx
        df[related_mv_cols] = df[related_mv_cols].div(df['circulation_mv'], axis=0) * (10 ** 8)
        df = df.fillna(0)

        # if not os.path.exists(f'{saving_path}{ticker}'): os.makedirs(f'{saving_path}{ticker}')
        # df.to_pickle(f'{saving_path}{ticker}/{date}.pkl')

        partition = df[factor_ret_cols].values.astype(np.float32)
        partition = np.pad(partition, ((63, 0), (0, 0)), 'constant')
        value_list.append(partition)
    #
    concat_value = np.concatenate(value_list, axis = 0)
    #
    ut.save_data_to_redis(rs, bytes(f'bilabels_{ticker}_{month}', encoding = 'utf-8'), concat_value)

    rs.close()

    return

def parallel_submit_ticker_monthly_numpy_train(db):
    """
    ??????????????????????????????????????????total_stock, total_tick, feature????????????numpy??????redis
    """
    date_ticker_dict = gen_date_ticker_dict(start_date = 20210506, end_date = 20210630)
    ticker_date_dict = rotate_key_value_monthly(date_ticker_dict)
    for month, ticker_dates in ticker_date_dict.items():
        Parallel(n_jobs=36, verbose=5, timeout=10000)(delayed(submit_train_data)(month, ticker, dates, db)
                                                      for ticker, dates in ticker_dates.items())
    return


if __name__ == "__main__":
    # move_files()
    parallel_submit_ticker_monthly_numpy_train(0)