import os
from collections import defaultdict
import numpy as np
from datetime import datetime
import utilities as ut
# import tst.utilities as ut
from joblib import Parallel,delayed
import logging
import pandas as pd
from tqdm import tqdm, trange
import src.logger as logger


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
                   'bid_inc', 'ask_inc2', 'bid_inc2', '10']


path = '/sgd-data/t0_data/500factor/500factors'


def gen_date_ticker_dict(start_date = 20210701, end_date = 20210931):
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
    Logger = logger.getLogger(name="Factor->Redis", level="INFO")
    data = pd.read_csv(file_path)
    # 原来如此……因子库里面的五十来个因子就是上面定义的名字
    data.columns = col_factors
    code = file_path.split('/')[-2]
    date = file_path.split('/')[-1][:-4]
    data.insert(2, 'code', code)
    data = ut.get_target(data, df_full_time)[factor_ret_cols].fillna(0)
    data = data.values.astype(np.float32)
    if len(data) == 0 or len(data) > 4800:
        return
    rs = ut.redis_connection(db)
    ut.save_data_to_redis(rs, b'numpy' + b'_' + bytes(f'{date}_{code}', encoding = 'utf-8'), data)
    Logger.info(f"Save factor for {code} at {date} to redis: finished")
    rs.close()
    return

def submit_concat_data(date, values, db):
    """
    shape: [stock_num, joint_tick_num, features]
    """
    rs = ut.redis_connection(db = db)

    counter = 0
    value_list = []
    for v in values:
        df = pd.read_csv(f'{path}/{v}/{date}.csv')
        value_list.append(df)
        if counter == 0:
            common_tick = set(df['timeidx'].tolist())
        else:
            common_tick = common_tick.intersection(set(df['timeidx'].tolist()))
        counter += 1
    try:
        value_list = list(map(lambda x: x.query('timeidx in @common_tick'), value_list))
    except:
        print('here')
    concat_value = np.concatenate(value_list, axis = 0)
    ut.save_data_to_redis(rs, f'{date}', concat_value)

    rs.close()
    return

def parallel_submit_concat_data(db):
    date_ticker_dict = gen_date_ticker_dict()
    Parallel(n_jobs=1, verbose=2, timeout=10000)(delayed(submit_concat_data)(date, values, db)
                                                  for date, values in date_ticker_dict.items())
    return



def submit_original_data(db, LOGGER):
    allpath, allname = ut.getallfile(path)
    df_full_time = ut.gen_df_full_time()

    LOGGER.info(f"Start saving factors to Redis...")

    # trade_days = rq.get_trading_dates(start_date=train_start_date, end_date=test_end_date)
    # trade_days_str = [(str(x).replace('-', '')) for x in trade_days]

    # train_file_path = [x for x in allpath if int(x.split('/')[-1][:-4]) <= train_end_date]
    file_path = [x for x in allpath if int(x.split('/')[-1][:-4])]

    # redis_keys = list(rs.keys())
    # cnn_redis_keys = [x for x in redis_keys if 'CNN' in str(x)]
    # train_redis_keys = [x for x in cnn_redis_keys if (int(str(x).split('_')[1]) <= train_end_date)
    #                     and (int(str(x).split('_')[1]) >= train_start_date)]
    # test_redis_keys = [x for x in cnn_redis_keys if (int(str(x).split('_')[1]) <= test_end_date)
    #                     and (int(str(x).split('_')[1]) >= test_start_date)]

    Parallel(n_jobs=48, timeout=10000)(delayed(gen_processed_data_to_redis)(file, df_full_time, db)
                                       for file in tqdm(file_path))

    return

if __name__ == "__main__":
    Logger = logger.getLogger(name="Factor->Redis", level="INFO")

    submit_original_data(db=0, LOGGER=Logger)

    # parallel_submit_concat_data(1)
    # rs = ut.redis_connection(db = 1)
    # all_keys = rs.keys()
    # for k in all_keys:
    #     k_df = ut.read_data_from_redis(rs, k)
    #     if len(k_df) > 5000:
    #         break