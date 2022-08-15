import json
from typing import Optional
from collections import deque

import numpy as np
from math import ceil
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
from functools import partial
import math
import train_dir_0.utilities as ut



factor_ret_cols = ['timeidx','price','vwp','spread','tick_spread','ref_ind_0','ref_ind_1','ask_weight_14',
                   'ask_weight_13','ask_weight_12','ask_weight_11','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
                   'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
                   'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
                   'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','bid_weight_11','bid_weight_12','bid_weight_13',
                   'bid_weight_14','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','turnover',
                   'cls_2', 'cls_5', 'cls_18']

def gen_mapping():
    cnt = 0
    classifier_mappping = {}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                classifier_mappping[(i, j, k)] = cnt
                cnt += 1
    return classifier_mappping

class HFDataset(Dataset):

    def __init__(self,
                 local_ids,
                 shard_dict,
                 batch_size,
                 seq_len = 50,
                 time_step = 3,
                 **kwargs):
        super().__init__(**kwargs)
        self.shard_dict = shard_dict
        self.seq_len = seq_len
        self._x = []
        self._y = []
        self.mapping = gen_mapping()

        self._load_array(local_ids, batch_size, seq_len, time_step)

    def _load_array(self, local_ids, batch_size, seq_len, time_step):
        # load from redis
        cnt = 0
        local_ids.sort()
        prev_key = '000000_00'
        data = None
        rs = ut.redis_connection(db = 0)
        # key_remain = []
        for idx in local_ids:
            key = self.shard_dict[idx]
            data = ut.read_data_from_redis(rs, key = key).astype(np.float32)
            data = torch.from_numpy(data)
            unfold_data = data.unfold(dimension=0, size=seq_len, step=time_step).transpose(1,2)
            batch_num = math.ceil(len(unfold_data) / batch_size)
            for i in range(1, batch_num + 1):
                if i == batch_num:
                    break

                self._x.append(unfold_data[(i - 1) * batch_size : i * batch_size, :, 1: -3])
                self._y.append(unfold_data[(i - 1) * batch_size: i * batch_size, -1, -3: ])

        rs.close()
        return


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self._x[idx], self._y[idx]

    def __len__(self):
        return len(self._x)

