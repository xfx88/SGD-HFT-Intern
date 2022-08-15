import torch
from joblib import Parallel, delayed
import os
import pandas as pd
import numpy as np
import rqdatac as rq
rq.init("15626436420", "vista2525")
import utilities as ut

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

related_mv_cols = ['ask_weight_14', 'ask_weight_13', 'ask_weight_12', 'ask_weight_11',
                   'ask_weight_10', 'ask_weight_9', 'ask_weight_8', 'ask_weight_7',
                   'ask_weight_6', 'ask_weight_5', 'ask_weight_4', 'ask_weight_3',
                   'ask_weight_2', 'ask_weight_1', 'ask_weight_0', 'bid_weight_0',
                   'bid_weight_1', 'bid_weight_2', 'bid_weight_3', 'bid_weight_4',
                   'bid_weight_5', 'bid_weight_6', 'bid_weight_7', 'bid_weight_8',
                   'bid_weight_9', 'bid_weight_10', 'bid_weight_11', 'bid_weight_12',
                   'bid_weight_13', 'bid_weight_14', 'ask_dec', 'bid_dec', 'ask_inc',
                   'bid_inc', 'ask_inc2', 'bid_inc2','turnover']

def gen_processed_data_to_redis(file_path, df_full_time):
    rs = ut.redis_connection()
    print(file_path)
    data = pd.read_csv(file_path)[col_factors]
    if len(data) == 0:
        return
    else:
        # data.columns = col_factors
        code = file_path.split('/')[-2]
        date = file_path.split('/')[-1][:-4]
        data.insert(2,'code',code)
        data = ut.get_target(data, df_full_time)
        ut.save_data_to_redis(rs, f'CNN_{date}_{code}',data)
        rs.close()
        return

def get_target(data, df_full_time):
  # data = read_data_from_redis(rs,key)
  df_1s = df_full_time.copy()
  data = data.set_index('time')
  df_1s.price = data.price
  df_1s.price = df_1s.price.fillna(method='ffill')
  time_periods = ['p_2', 'p_5', 'p_18']
  for time_period in time_periods:
    df_1s[time_period] = df_1s.price.shift(-int(time_period.split('_')[-1]) * 3) / df_1s.price - 1
  data[time_periods] = df_1s[time_periods]
  data['p_diff'] = data['p_5'] - data['p_2']
  return data

def gen_df_full_time():
  full_time = list(pd.date_range(start='09:30:00', end='11:30:00', freq='S'))
  full_time.extend(list(pd.date_range(start='13:00:00', end='15:00:00', freq='S')))
  full_time = [str(x.time()) for x in full_time]
  df_full_time = pd.DataFrame(index=full_time, columns={'price'})
  return df_full_time

allpath = []
allname = []
def getallfile(path, allfile, allname):
    """
    递归获取文件夹下所有文件
    :param path: 路径
    :param allfile: 所有文件
    :param allname: 所有文件名
    :return: allfile, allname
    """
    filelist = []
    filelist = os.listdir(path)
    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            getallfile(filepath, allfile, allname)
        else:
            allfile.append(filepath)
            allname.append(filename)
    return allfile, allname

def get_float_market_values(start_date , end_date):
    rs = ut.redis_connection()
    redis_keys = list(rs.keys())
    cnn_redis_keys = [x for x in redis_keys if 'CNN' in str(x)]
    all_keys = [x for x in cnn_redis_keys if (int(str(x).split('_')[1]) <= end_date)
                and (int(str(x).split('_')[1]) >= start_date)]
    codes = list(set([x.decode('utf-8').split('_')[-1] for x in all_keys]))
    order_book_ids = []
    for code in codes:
        if code.startswith('6'):
            order_book_ids.append(code + '.XSHG')
        else:
            order_book_ids.append(code + '.XSHE')
    df_float_market =  rq.get_factor(order_book_ids, 'a_share_market_val_in_circulation',
                                     start_date = start_date, end_date = end_date)
    df_float_market = df_float_market.reset_index()
    df_float_market = df_float_market.rename(columns = {'order_book_id' : 'code'})
    df_float_market.code = df_float_market.code.apply(lambda x : x[:6])
    df_float_market.date = df_float_market.date.apply(lambda x : str(x.date()).replace('-',''))
    df_float_market = df_float_market.set_index(['code','date'])
    def load_market_values_to_redis(key , df_float_market):
        rs  = ut.redis_connection()
        date , code = key.decode('utf-8').split('_')[-2:]
        data_ori = ut.read_data_from_redis(rs,key)
        # del data_ori['circulation _market_value']
        data_ori['circulation_mv'] = df_float_market[df_float_market.index == (code,date)].a_share_market_val_in_circulation.iloc[0]
        ut.save_data_to_redis(rs, key.decode('utf-8'), data_ori)
        rs.close()
        return
    Parallel(n_jobs=36,verbose=2,timeout=100000)(delayed(load_market_values_to_redis)(key,df_float_market)
                                                  for key in all_keys)

def get_samples(rs, key, tick_nums):
    data = ut.read_data_from_redis(rs, key)
    data[factor_ret_cols] = data[factor_ret_cols].fillna(0)
    data[['price','vwp']] = (data[['price','vwp']].div(data.preclose, axis = 0) - 1) * 1000
    data['timeidx'] = (data.timeidx - 7114) / 4100 # normalize timeidx
    data[related_mv_cols] = data[related_mv_cols].div(data['circulation_mv'] ,  axis = 0) * (10 ** 11)
    data = data[factor_ret_cols]

    # for col in predict_cols:
    #     data[col] = data[col]
        # data[col] = data[col] * 1000
    data = data.values.astype(np.float32)
    if len(data) < tick_nums:
        return
    zero_tensor = torch.zeros((tick_nums - 1, len(factor_ret_cols)))
    data_t = torch.cat((zero_tensor, torch.tensor(data)))
    data2 = data_t.unfold(0, tick_nums, 2).unsqueeze(dim = 3)
    return data2