import os
from collections import defaultdict, deque

import numpy as np
import pickle

from datetime import datetime
from functools import partial

import utilities as ut
from joblib import Parallel,delayed
import pandas as pd
import rqdatac as rq
from tqdm import tqdm
rq.init(15626436420, 'vista2525')

col_factors = ['date','code','timeidx','price','vwp','ask_price','bid_price','ask_price2','bid_price2','ask_price4',
               'bid_price4','ask_price8','bid_price8','spread','tick_spread','ref_ind_0','ref_ind_1','ask_weight_14',
               'ask_weight_13','ask_weight_12','ask_weight_11','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
               'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
               'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
               'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','bid_weight_11','bid_weight_12','bid_weight_13',
               'bid_weight_14','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','preclose','limit','turnover',
               'circulation_mv', 'p_2','p_5','p_18','p_diff']

factor_ret_cols = ['timeidx','price_pct','vwp_pct','ref_ind_0','ref_ind_1','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
                   'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
                   'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
                   'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','turnover',
                   'cls_2', 'cls_5', 'cls_18', 'subcls_2', 'subcls_5', 'subcls_18', 'p_2','p_5','p_18']

# factor_ret_cols = ['timeidx','price_pct','vwp_pct','spread','tick_spread','ref_ind_0','ref_ind_1','ask_weight_14',
#                    'ask_weight_13','ask_weight_12','ask_weight_11','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
#                    'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
#                    'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
#                    'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','bid_weight_11','bid_weight_12','bid_weight_13',
#                    'bid_weight_14','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','turnover',
#                    'cls_2', 'cls_5', 'cls_18', 'subcls_2', 'subcls_5', 'subcls_18', 'p_2','p_5','p_18']

tick_cols =['ask_price1', 'ask_price2', 'ask_price3',  'ask_price4', 'ask_price5',
            'bid_price1',  'bid_price2','bid_price3', 'bid_price4','bid_price5',
            'ask_volume1', 'ask_volume2','ask_volume3','ask_volume4', 'ask_volume5',
            'bid_volume1', 'bid_volume2', 'bid_volume3', 'bid_volume4', 'bid_volume5']

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
    classifier_mapping = {}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                classifier_mapping[(i, j, k)] = cnt
                cnt += 1
    return classifier_mapping

MAPPING = gen_mapping()

def judger(x, threshold):
    if x > threshold:
        return 2
    elif x < -threshold:
        return 1
    else:
        return 0

def judger_to5(x, threshold1, threshold2):

    if x >= threshold2:
        return 4
    elif x <= -threshold2:
        return 2
    elif x >= threshold1 and x < threshold2:
        return 3
    elif x > -threshold2 and x <= -threshold1:
        return 1

    return 0


path = '/home/yby/SGD-HFT-Intern/Projects/T0/Data/'
tgt_path = '/home/yby/SGD-HFT-Intern/Projects/T0/Data2/'
saving_path = '/home/yby/SGD-HFT-Intern/Projects/T0/Data_labels/'

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

def draw_stats(month, ticker, values):
    value_list = []

    values.sort()
    for date in values:
        # 过滤掉无效长度的数据
        df = pd.read_pickle(f'{path}{ticker}/{date}.pkl').reset_index()

        if len(df) < 10: continue
        if len(df) > 4800: continue

        value_list.append(df[['ref_ind_0','ref_ind_1','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
                              'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1',
                              'ask_weight_0','bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4',
                              'bid_weight_5','bid_weight_6','bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10',
                              'ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','turnover']].values)

    concat_value = np.concatenate(value_list, axis=0)
    concat_mean = (concat_value.mean(axis = 0)).tolist()
    concat_std = (concat_value.std(axis=0) / np.sqrt(len(value_list))).tolist()
    stat_dict[month][ticker]["mean"] = concat_mean
    stat_dict[month][ticker]["std"] = concat_std
    return

def parallel_submit_statdict(db):
    """
    用于按时间读取数据，转换成（total_stock, total_tick, feature）形状的numpy存入redis
    """
    global stat_dict
    stat_dict = defaultdict(partial(defaultdict, partial(defaultdict, float)))
    date_ticker_dict = gen_date_ticker_dict()
    ticker_date_dict = rotate_key_value_monthly(date_ticker_dict)
    for month, ticker_dates in tqdm(ticker_date_dict.items()):
        for ticker, dates in ticker_dates.items():
            draw_stats(month, ticker, dates)

    with open("/home/yby/SGD-HFT-Intern/Projects/T0/CNN/data_submit_progs/statDict_for_submit.pkl", "wb") as f:
        pickle.dump(stat_dict, f)

    return


def submit_train_data(month, ticker, values, db, stat_dict):
    """
    shape: [stock_num, joint_tick_num, features]
    """

    value_list = []
    rs = ut.redis_connection(db=db)

    prev_month_1 = str(int(month) - 2)
    prev_month_2 = str(int(month) - 1)
    prev_month_1 = "0" + prev_month_1 if len(prev_month_1) == 1 else prev_month_1
    prev_month_2 = "0" + prev_month_2 if len(prev_month_2) == 1 else prev_month_2
    # mean_1 = np.array(stat_dict[prev_month_1][ticker]["mean"])
    # mean_2 = np.array(stat_dict[prev_month_2][ticker]["mean"])
    # mean_now = ((mean_1 + mean_2) / 2).tolist()
    std_1 = np.array(stat_dict[prev_month_1][ticker]["std"])
    std_2 = np.array(stat_dict[prev_month_2][ticker]["std"])
    std_now = np.sqrt(std_1 ** 2 + std_2 ** 2 + 2 * std_1 * std_2).tolist()

    values.sort()
    for date in values:
        # 过滤掉无效长度的数据
        df = pd.read_pickle(f'{path}{ticker}/{date}.pkl').reset_index()

        if len(df) < 10: continue
        if len(df) > 4800: continue



        df['date_time'] = df.apply(lambda x: str(x['date']) + ' ' + x['time'], axis = 1)
        df['date_time'] = pd.to_datetime(df['date_time'])

        df.set_index('date_time', inplace=True)

        df[['price_pct', 'vwp_pct']] = (df[['price', 'vwp']].div(df.preclose, axis=0) - 1) * 1000
        df['timeidx'] = (df.timeidx- 7114) / 4100  # normalize timeidx
        df[related_mv_cols] = df[related_mv_cols].div(df['circulation_mv'], axis=0) * (10 ** 8)
        df = df.fillna(0)

        df["p_2"] = (df["p_2"]) / std_now[0]
        df["p_5"] = (df["p_5"]) / std_now[1]
        df["p_18"] = (df["p_18"]) / std_now[2]

        # cls这里对应的为三分类问题
        df['cls_2'] = df['p_2'].apply(lambda x: judger(x, 1.645))
        df['cls_5'] = df['p_5'].apply(lambda x: judger(x, 1.645))
        df['cls_18'] = df['p_18'].apply(lambda x: judger(x, 1.645))

        # 这里subcls对应的是五分类问题
        df['subcls_2'] = df['p_2'].apply(lambda x: judger_to5(x, 1.3, 2.55))
        df['subcls_5'] = df['p_5'].apply(lambda x: judger_to5(x, 1.3, 2.55))
        df['subcls_18'] = df['p_18'].apply(lambda x: judger_to5(x, 1.3, 2.55))

        # if month == "11":
        #     if not os.path.exists(f'{saving_path}{ticker}'): os.makedirs(f'{saving_path}{ticker}')
        #     df.to_pickle(f'{saving_path}{ticker}/{date}.pkl')

        partition = df[factor_ret_cols].values.astype(np.float32)
        partition = np.pad(partition, ((63, 0), (0, 0)), 'constant')
        value_list.append(partition)
    #
    concat_value = np.concatenate(value_list, axis = 0)
    # 决定是否存储进redis，标签为distlabels
    ut.save_data_to_redis(rs, bytes(f'distlabels_{ticker}_{month}', encoding = 'utf-8'), concat_value)

    rs.close()

    return

def parallel_submit_ticker_monthly_numpy_train(db):
    """
    用于按时间读取数据，转换成（total_stock, total_tick, feature）形状的numpy存入redis
    """

    stat_dict = dict()
    temp_dict = pickle.load(open("/home/yby/SGD-HFT-Intern/Projects/T0/CNN/data_submit_progs/statDict_for_submit.pkl", "rb"))
    for m, t in temp_dict.items():
        try:
            stat_dict[m]
        except:
            stat_dict[m] = dict()

        for ticker, d in t.items():
            try:
                stat_dict[m][ticker]
            except:
                stat_dict[m][ticker] = dict()

            for k, v in d.items():
                stat_dict[m][ticker][k] = v



    date_ticker_dict = gen_date_ticker_dict()
    ticker_date_dict = rotate_key_value_monthly(date_ticker_dict)
    for month, ticker_dates in ticker_date_dict.items():
        if month != '05' and month != '06':
            Parallel(n_jobs=48, verbose=5, timeout=10000)(delayed(submit_train_data)(month, ticker, dates, db, stat_dict.copy())
                                                          for ticker, dates in tqdm(ticker_dates.items()))
    return


if __name__ == "__main__":
    # parallel_submit_statdict(0)
    parallel_submit_ticker_monthly_numpy_train(0)