import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from datetime import datetime
from datetime import timedelta
import os
import glob
import gc
import dask
import joblib
from multiprocessing import cpu_count
import functools
from itertools import product
from collections import OrderedDict, defaultdict, Counter
import warnings
import hfhd.hf as hf
from tqdm import tqdm, trange
import pickle
import redis
from RemoteQuery import *
from cal_features import *

warnings.filterwarnings("ignore")
pd.set_option('expand_frame_repr', False)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 180)
pd.set_option('display.max_columns', None)


# GLOBAL VARIABLES
SRC = RemoteSrc()
POOL = redis.ConnectionPool(host='localhost', port=6379, decode_responses=False)
SERVER = redis.Redis(connection_pool=POOL)

# FUNCTIONS
def get_stock_list(date_list):
    """
    return: the stock list in which the stock has data for all the dates in the date list
    """
    SRC = RemoteSrc()
    stock_list_final = []
    for date in date_list:
        code = SRC.get_stock_list(date)
        stock_list = pd.DataFrame(code, columns=['code'])

        cond1 = (stock_list['code'].str.startswith('0')) & (stock_list['code'].str.contains('SZSE'))
        cond2 = (stock_list['code'].str.startswith('6')) & (stock_list['code'].str.contains('SSE'))
        cond3 = (stock_list['code'].str.startswith('30')) & (stock_list['code'].str.contains('SZSE'))

        # exist duplicated number for index and stock
        stock_list.drop_duplicates(inplace=True)

        stock_list = stock_list[cond1 | cond2 | cond3]
        stock_list['code'] = stock_list['code'].str[:6]
        stock_list = sorted(stock_list['code'].to_list())
        if stock_list_final.__len__() == 0:
            stock_list_final = stock_list
        else:
            # pick intersection
            # stock_list_final = list(set(stock_list_final).intersection(set(stock_list)))
            stock_list_final = np.intersect1d(stock_list_final, stock_list).tolist()
    print("Total number of stocks: %s"%stock_list_final.__len__())
    stock_list_final.sort()
    return stock_list_final

def get_tick(code, date) -> pd.DataFrame:
    asset = SRC.get_raw_bars(code, date)
    begin_time = '09:30:00'
    end_time = '14:57:00'
    range_ = (asset['time'] >= begin_time)&(asset['time'] < end_time)
    asset = asset[range_].reset_index()
    asset['delta_quote'] = asset['midquote'].diff()

    # NOTE: add if using the tick time for prediction
    # use tick time only
    # cond1 = asset['delta_quote'] != 0
    # cond2 = np.abs(asset['delta_quote']) < 1
    # asset = asset[cond1 & cond2]


    # calculate necessary features
    asset['midquote'] = np.log((asset['ask_price1'] + asset['bid_price1'])/2) # log midquote price
    asset['wpr'] = calc_wap1(asset)
    asset['wpr_ret'] = log_return(asset['wpr'])
    asset['rel_vol'] = realized_volatility(asset['wpr_ret'])


    return asset

def get_ticks(code, date_list) -> pd.DataFrame:
    try:
        res = list(map(lambda date: get_tick(code, date), date_list))
        asset_combined = pd.concat(res, ignore_index=True)
        asset_combined.sort_values(by=['server_time'], ascending=True, inplace=True)
        return asset_combined
    except:
        print("Fail to retrive the data for %s"%(code))
        return None

def process_one_stock(code, date) -> pd.Series:
    try:
        asset = SRC.get_raw_bars(code, date)
        begin_time = '09:30:00'
        end_time = '14:57:00'
        range_ = (asset['time'] >= begin_time) & (asset['time'] < end_time)
        asset = asset[range_].reset_index()
        asset['midquote'] = np.log((asset['ask_price1'] + asset['bid_price1'])/2)

        # delete the data if limit-up or down happens
        if asset[(asset['ask_price1'] == 0) | (asset['bid_price1'] == 0)].shape[0] > 0:
            print(f"limit up/down occurs at {date} for stock {code}")
            return pd.Series([np.nan]*asset.shape[0], index=asset["server_time"].tolist())

        # NOTE: use original time or tick time
        # only need the server time and logarithmic price for calculating HY function
        asset = asset.set_index("server_time")['midquote']

        # asset['delta_quote'] = asset['midquote'].diff()
        # asset = asset.loc[asset['delta_quote'] != 0].set_index("server_time")['midquote']
        return asset
        
    except:
        print("Fail to retrive the data for %s on %s"%(code, date))
        return pd.Series([np.nan])

# NOTE: save the data into redis to speed up
def preprocess(code_list, date) -> list:
    df_bytes_from_redis = SERVER.hget("tick_list", f"{date}")
    if (df_bytes_from_redis is None):
        # one_day_data = list(map(lambda cd: process_one_stock(cd, date=date), code_list))
        f_par = functools.partial(process_one_stock, date=date)
        one_day_data = joblib.Parallel(n_jobs=50)(joblib.delayed(f_par)(code) for code in code_list)
        df_bytes = pickle.dumps(one_day_data)
        SERVER.hset("tick_list", f'{date}', df_bytes)
    else:
        one_day_data = pickle.loads(df_bytes_from_redis)
    return one_day_data

def prepare_all_data(code_list, date_list:list) -> list:
    f_par = functools.partial(preprocess, code_list=code_list)
    all_stock_data = list(map(lambda date: f_par(date=date), date_list))
    # all_stock_data = joblib.Parallel(n_jobs=date_list.__len__())(joblib.delayed(f_par)(date=date) for date in date_list)
    return all_stock_data

def calc_lag(df, lag) -> pd.DataFrame:
    df.index = df.index - timedelta(seconds=lag) if lag >= 0 else df.index + timedelta(seconds=(-lag))
    return df

def calc_pairwise_HY(lag, one_day_data) -> list:
    format_data = [one_day_data[0]] + list(map(lambda df: calc_lag(df, lag=lag), one_day_data[1:]))
    pairwise_hy_list = hf.hayashi_yoshida_simplified(format_data, theta=None, k=None, choice='corr')
    return pairwise_hy_list.tolist()

def calc_all_HY_oneday(date):
    df_bytes_from_redis = SERVER.hget("tick_list", date)
    test_df = pickle.loads(df_bytes_from_redis)
    hy = hf.hayashi_yoshida(test_df, choice='cov')
    hy[hy == 0] = np.nan
    return hy

def calc_all_HY(date_list):
    with dask.config.set(scheduler='processes', num_workers=min(48, date_list.__len__())):
        corr_hy = dask.compute(dask.delayed(calc_all_HY_oneday)(date) for date in date_list)[0]
    mean_hy = np.nanmean(corr_hy, axis=0)
    return mean_hy

def calc_stats(code, stock_data, date_list, nums):
    whole_nums = list(map(lambda x: -x, nums))[::-1] + [0] + nums
    target = list(map(lambda dt: process_one_stock(code, dt), date_list))
    # TODO check correct or not
    meg_data = np.column_stack((target, stock_data)).tolist()
    # TODO determine the method for parallel computing (parallel w.r.t. date or lag) and the tools (iterator, parallel, dask, etc.)
    stats = []
    for data in meg_data:
        # temp = list(map(lambda lag: calc_pairwise_HY(lag, data), whole_nums))
        temp = joblib.Parallel(n_jobs=whole_nums.__len__())(joblib.delayed(functools.partial(calc_pairwise_HY, one_day_data=data))(lag) for lag in whole_nums)
        stats.append(temp)
    # with dask.config.set(scheduler='processes', num_workers=50):
        # result = dask.compute(dask.delayed(f_par)(code) for code in code_list)[0]
    mean_stats = np.nanmean(stats, axis=0) # average towards the day, the shape is (lag_periods, assets_num)  
    std_stats = np.nanstd(stats, axis=0)
    print(f"Finish calculating HY correlation for {code}")
    return mean_stats, std_stats

def calc_all_pairs(code_list1, code_list2, date_list, nums):
    all_stock_data = prepare_all_data(code_list2, date_list)
    result = list(map(lambda code: calc_stats(code, all_stock_data, date_list, nums), code_list1))

    # g_par = functools.partial(calc_stats, stock_data=all_stock_data, date_list=date_list, nums=nums)
    # result = joblib.Parallel(n_jobs=code_list1.__len__())(joblib.delayed(g_par)(code) for code in code_list1)
    # with dask.config.set(scheduler='processes', num_workers=code_list1.__len__()):
        # result = dask.compute(dask.delayed(g_par)(code) for code in code_list1)[0]

    result = OrderedDict(zip(code_list1, result))
    return result