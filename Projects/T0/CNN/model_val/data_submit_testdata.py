import os
from collections import OrderedDict, defaultdict

import numpy as np

from datetime import datetime

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

factor_ret_cols = ['timeidx','price','vwp','ask_price','bid_price','ask_price2','bid_price2','ask_price4',
                   'bid_price4','ask_price8','bid_price8','spread','tick_spread','ref_ind_0','ref_ind_1','ask_weight_14',
                   'ask_weight_13','ask_weight_12','ask_weight_11','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
                   'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
                   'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
                   'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','bid_weight_11','bid_weight_12','bid_weight_13',
                   'bid_weight_14','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','preclose','limit','turnover',
                   'circulation_mv', 'p_2','p_5','p_18','p_diff']


path = "/home/yby/SGD-HFT-Intern/Projects/T0/Data"


def gen_date_ticker_dict(start_date = 20211101, end_date = 20211130):
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


def test_data_submit(date, values, db):
    df_full_time = ut.gen_df_full_time()
    date_stock_index_dict = OrderedDict()

    maxlen = 4800
    values.sort()
    df_list = []
    stock_list = [] # 存储过滤后的股票代码

    rs = ut.redis_connection(db=db)
    for v in values:
        # 过滤掉无效长度的数据
        df = pd.read_pickle(f'{path}/{v}/{date}.pkl').fillna(0)
        if len(df) < 10: continue
        # df.columns = col_factors
        # df = ut.get_target(df, df_full_time).fillna(0)
        if len(v) > 4800: continue
        ut.save_data_to_redis(rs, bytes(f'df_{date}_{v}', encoding='utf-8'), df)

        # v_time_list = df.index.tolist()
        # v_time_list.sort()
        # date_stock_index_dict[v] = v_time_list
        # df = df[factor_ret_cols]
        # df_list.append(df.values)
    #
    #
    # dfs = []
    # #
    # for v in df_list:
    #     if not len(v) >= 4800:
    #         v = np.pad(v, ((0, maxlen - len(v)), (0, 0)), mode='constant')
    #         v = v.reshape((1, -1, v.shape[-1]))
    #     dfs.append(v.astype(np.float32))
    # try:
    #     concat_value = np.concatenate(dfs, axis=0)
    # except:
    #     print("here")

    # ut.save_data_to_redis(rs, bytes(f'numpy_{date}', encoding='utf-8'), concat_value)
    # ut.save_data_to_redis(rs, bytes(f'stock_index_{date}', encoding='utf-8'), date_stock_index_dict)
    rs.close()

    return

def parallel_submit_date_numpy_test(db):
    """
    用于按时间读取数据，转换成（total_stock, total_tick, feature）形状的numpy存入redis
    """
    date_ticker_dict = gen_date_ticker_dict(start_date = 20211111, end_date = 20211130)
    Parallel(n_jobs=16, verbose=2, timeout=10000)(delayed(test_data_submit)(date, values, db)
                                                  for date, values in date_ticker_dict.items())
    return

if __name__ == "__main__":
    parallel_submit_date_numpy_test(0)