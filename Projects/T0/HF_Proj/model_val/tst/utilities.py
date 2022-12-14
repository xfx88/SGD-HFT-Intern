import datetime
import pandas as pd
import torch
import numpy as np

import os
from dataclasses import dataclass

import redis
import pickle
from pympler import asizeof
import torch.nn.functional as F
from tst.settings import REDIS_HOST,REDIS_PORT,REDIS_PASSWORD

allpath = []
allname = []

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
    attention_size: int or None = None # Attention window size
    window: int = None
    d_input: int = 0 # From dataset
    d_output: int = 0
    padding: int = None
    dropout: float = 0.3  # Dropout rate
    pe: str = None  # Positional encoding
    chunk_mode: str = None




def getallfile(path):
    allfilelist = os.listdir(path)
    # ????????????????????????????????????????????????
    for file in allfilelist:
        filepath = os.path.join(path, file)
        # ???????????????????????????????????????
        if os.path.isdir(filepath):
            getallfile(filepath)
        # ??????????????????????????????????????????????????????
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
    ??????redis??????
    @param db: ?????????ID
    @return: redis????????????
    """
    return redis.StrictRedis(
        host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db = db)

def save_data_to_redis(rs, key, df):
    """
    ???????????????redis
    @param rs: redis connection
    @param key: redis key
    @param df: dataframe ??????
    """
    df_bytes = pickle.dumps(df)
    return rs.set(key, df_bytes)


def read_data_from_redis(rs, key):
    """
    ???Redis???????????????
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
    ????????????
    @param obj: ????????????
    @param file_name: ??????
    """
    f = open(file_name, "wb")
    pickle.dump(obj, f)
    f.close()
