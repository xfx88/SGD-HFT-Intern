import os
import shutil
from collections import defaultdict, deque

import numpy as np

from datetime import datetime
from functools import partial
import paramiko
from tqdm import tqdm

import utilities as ut
from joblib import Parallel,delayed
import pandas as pd
import rqdatac as rq
rq.init(15626436420, 'vista2525')

col_factors = ['date','code','timeidx','price','vwp','ask_price','bid_price','ask_price2','bid_price2','ask_price4',
               'bid_price4','ask_price8','bid_price8','spread','ref_ind_0','ref_ind_1','ask_weight_14',
               'ask_weight_13','ask_weight_12','ask_weight_11','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
               'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
               'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
               'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','bid_weight_11','bid_weight_12','bid_weight_13',
               'bid_weight_14','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','preclose','limit','turnover',
               'circulation_mv', 'p_2','p_5','p_18','p_diff']

factor_cols = ["vwp", 'price', 'timeidx','price_pct','vwp_pct', 'currentRet','spread','tick_spread','ref_ind_0','ref_ind_1','ask_weight_14',
               'ask_weight_13','ask_weight_12','ask_weight_11','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
               'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
               'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
               'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','bid_weight_11','bid_weight_12','bid_weight_13',
               'bid_weight_14','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','turnover']
# factor_ret_cols = ['timeidx','price_pct','vwp_pct','spread','tick_spread','ref_ind_0','ref_ind_1','ask_weight_14',
#                    'ask_weight_13','ask_weight_12','ask_weight_11','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
#                    'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
#                    'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
#                    'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','bid_weight_11','bid_weight_12','bid_weight_13',
#                    'bid_weight_14','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','turnover',
#                    'cls_2', 'cls_5', 'cls_18', 'subcls_2', 'subcls_5', 'subcls_18', 'p_2','p_5','p_18']


tick_cols = ["open", "high", "low", "net_vol"]

pred_cols = ['subcls_2', 'subcls_5', 'subcls_18', 'subcls_diff', 'p_2','p_5','p_18','p_diff']

related_mv_cols = ['ask_weight_14','ask_weight_13','ask_weight_12','ask_weight_11','ask_weight_10','ask_weight_9',
                   'ask_weight_8','ask_weight_7', 'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3',
                   'ask_weight_2','ask_weight_1','ask_weight_0', 'bid_weight_0','bid_weight_1','bid_weight_2',
                   'bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6', 'bid_weight_7','bid_weight_8',
                   'bid_weight_9','bid_weight_10','bid_weight_11','bid_weight_12','bid_weight_13', 'bid_weight_14',
                   'ask_dec', 'bid_dec', 'ask_inc', 'bid_inc', 'ask_inc2', 'bid_inc2','turnover']

class RemoteSrc:
    REMOTE_PATH = "/sgd-data/data/stock/"
    TEMP = "/home/yby/SGD-HFT-Intern/Projects/T0/tickdata_temp/"

    def __init__(self):
        if not os.path.exists(self.TEMP):
            os.mkdir(self.TEMP)
        self.dict_stocksPerDay = defaultdict(list)

    def get_raw_bars(self, ticker, date):

        local_path = f"{self.TEMP}{ticker}/{ticker}_{date}.csv.gz"

        data = pd.read_csv(local_path)
        data['server_time'] = pd.to_datetime(data.server_time)
        data['time'] = data.apply(lambda x: str(x['server_time'].time()), axis = 1)

        return data

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
    elif x > threshold1 and x < threshold2:
        return 3
    elif x > -threshold2 and x < -threshold1:
        return 1

    return 0


path = '/home/yby/SGD-HFT-Intern/Projects/T0/Data/'
tgt_path = '/home/yby/SGD-HFT-Intern/Projects/T0/Data2/'
saving_path = '/home/yby/SGD-HFT-Intern/Projects/T0/Data_labels/'
stat_dict = defaultdict(partial(defaultdict, float))


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
    source = RemoteSrc()

    value_list = []
    rs = ut.redis_connection(db=db)

    values.sort()
    for date in values:
        # 过滤掉无效长度的数据
        raw_ticks = source.get_raw_bars(ticker, date).set_index("time")
        raw_ticks["net_vol"] = raw_ticks["volume"].diff()
        raw_ticks = raw_ticks[tick_cols + ["last", "preclose"]]
        raw_ticks["last"] = raw_ticks["last"].fillna("ffill")
        raw_ticks = raw_ticks.fillna(0)

        df = pd.read_pickle(f'{path}{ticker}/{date}.pkl').reset_index()

        if len(df) < 10: continue
        if len(df) > 4800: continue

        df['date_time'] = df.apply(lambda x: str(x['date']) + ' ' + x['time'], axis = 1)
        df['date_time'] = pd.to_datetime(df['date_time'])
        df.set_index("time", inplace = True)

        df[["price_pct", "vwp_pct"]] = (df[["price", "vwp"]].div(df["price"].shift(1), axis = 0) - 1) * 1e3
        # df[['price_pct']] = (df[['price']].pct_change()) * 1e3
        df['timeidx'] = (df.timeidx - 7114) / 4100  # normalize timeidx
        df["currentRet"] = (df["vwp"] / df["preclose"] - 1) * 1e3

        df[related_mv_cols] = df[related_mv_cols] / 100
        # df["p_diff"] = df["p_18"] - df["p_5"]

        df['subcls_2'] = df['p_2'].apply(lambda x: judger_to5(x, 0.0005, 0.0015))
        df['subcls_5'] = df['p_5'].apply(lambda x: judger_to5(x, 0.001, 0.002))
        df['subcls_18'] = df['p_18'].apply(lambda x: judger_to5(x, 0.0015, 0.003))

        def judge_inverse(x, threshold1, threshold2):
            # 反转预测，用于开仓
            if x["p_diff"] >= threshold2 and x["subcls_2"] > 0 and x["subcls_2"] < 3:
                if x["subcls_5"] == 4:
                    return 4
                return 3

            # elif x["p_diff"] >= threshold1 and x["p_diff"] < threshold2:
            #     if (x["subcls_2"] <= 1) and x["subcls_5"] == 3:
            #         return 3

            elif x["p_diff"] <= -threshold2 and x["subcls_2"] >= 3:
                if  x["subcls_5"] == 2:
                    return 2
                return 1

            # elif x["p_diff"] > -threshold2 and x["p_diff"] <= -threshold1:
            #     if (x["subcls_2"] >= 3) and x["subcls_5"] == 1:
            #         return 1

            return 0

        df["subcls_diff"] = df.apply(lambda x: judge_inverse(x, 0., 0.002), axis = 1)

        df = pd.merge(df[factor_cols + pred_cols + ["date_time", "circulation_mv"]], raw_ticks, how = "left", left_index = True, right_index=True)
        df["hl_spread"] = df["high"] / df["low"] - 1
        df["net_vol"] = df["net_vol"] / (df["circulation_mv"] / df["preclose"]) * 1e6

        df.reset_index(inplace=True)
        df.set_index('date_time', inplace=True)
        df = df.fillna(0)

        if np.argwhere(df.isna().values).sum() or np.argwhere(df.values == np.inf).sum():
            print("here")


        if month != "11":
            partition = df[factor_cols + ["hl_spread", "net_vol"] + pred_cols].values.astype(np.float32)
            partition = partition[: -1] if len(partition) % 2 == 1 else partition
            partition = np.pad(partition, ((63, 0), (0, 0)), 'constant')
            value_list.append(partition)
        else:
            if not os.path.exists(f'{saving_path}{ticker}'): os.makedirs(f'{saving_path}{ticker}')
            df.to_pickle(f'{saving_path}{ticker}/{date}.pkl')

    if month != "11":
        concat_value = np.concatenate(value_list, axis = 0)
        ut.save_data_to_redis(rs, bytes(f'manulabels_{month}_{ticker}', encoding='utf-8'), concat_value)

    rs.close()

    return

def parallel_submit_ticker_monthly_numpy_train(db):
    """
    用于按时间读取数据，转换成（total_stock, total_tick, feature）形状的numpy存入redis
    """

    date_ticker_dict = gen_date_ticker_dict()
    ticker_date_dict = rotate_key_value_monthly(date_ticker_dict)
    for month, ticker_dates in ticker_date_dict.items():
        Parallel(n_jobs=48, verbose=0)(delayed(submit_train_data)(month, ticker, dates, db)
                                      for ticker, dates in tqdm(ticker_dates.items()))
    return


if __name__ == "__main__":
    # move_files()
    parallel_submit_ticker_monthly_numpy_train(0)