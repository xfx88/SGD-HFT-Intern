import datetime
import pandas as pd
import torch
import numpy as np
from collections import OrderedDict

import os
from dataclasses import dataclass
from torch.utils.data import random_split

import redis
import pickle
from pympler import asizeof
import torch.nn.functional as F
from settings import REDIS_HOST,REDIS_PORT,REDIS_PASSWORD


allpath = []
allname = []

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
@dataclass
class TrainingOptions:
    EPOCHS: int
    BATCH_SIZE: int
    NUM_WORKERS: int
    LR: float

    # Model parameters
    d_model: int  # Latent dim
    query: int  # Query size
    value: int  # Value size
    heads: int  # Number of heads
    N_stack: int  # Number of encoder and decoder to stack
    d_input: int  # From dataset
    d_output: int
    attention_size: int or None = None # Attention window size
    window: int = None
    padding: int = None
    dropout: float = 0.15  # Dropout rate
    pe: str = None  # Positional encoding
    chunk_mode: str or None = None


def train_val_splitter(dataset, epoch_idx, percent = 0.85, validation = True):
    len_train = int(len(dataset) * percent)
    # train_set, val_set = random_split(dataset, [len_train, len_valid], generator=torch.Generator().manual_seed(epoch_idx))
    train_set, val_set = dataset[:len_train], dataset[len_train:]
    if not validation:
        del val_set
        return train_set, None
    return train_set, val_set

def ddpModel_to_normal(ddp_state_dict: OrderedDict):
    new_state_dict = OrderedDict()
    for k, v in ddp_state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict



def getallfile(path):
    allfilelist = os.listdir(path)
    # 遍历该文件夹下的所有目录或者文件
    for file in allfilelist:
        filepath = os.path.join(path, file)
        # 如果是文件夹，递归调用函数
        if os.path.isdir(filepath):
            getallfile(filepath)
        # 如果不是文件夹，保存文件路径及文件名
        elif os.path.isfile(filepath):
            allpath.append(filepath)
            allname.append(file)
    return allpath, allname


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
  data[time_periods] = df_1s[time_periods]
  return data


def redis_connection(db = 0):
    """
    返回redis连接
    @param db: 数据库ID
    @return: redis连接对象
    """
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db = db, health_check_interval=30)

def save_data_to_redis(rs, key, df):
    """
    保存数据到redis
    @param rs: redis connection
    @param key: redis key
    @param df: dataframe 数据
    """
    df_bytes = pickle.dumps(df)
    return rs.set(key, df_bytes)


def read_data_from_redis(rs, key):
    """
    从Redis中获取数据
    @param rs: redis connection
    @param key: redis key
    @return: Dataframe
    """
    df_bytes_from_redis = rs.get(key)
    if not df_bytes_from_redis:
        return None
    df_from_redis = pickle.loads(df_bytes_from_redis)
    return df_from_redis

def dump_data(obj, file_name):
    """
    保存数据
    @param obj: 数据对象
    @param file_name: 目录
    """
    f = open(file_name, "wb")
    pickle.dump(obj, f)
    f.close()
