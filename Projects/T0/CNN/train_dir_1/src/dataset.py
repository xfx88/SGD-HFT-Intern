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
import train_dir_1.utilities as ut



factor_ret_cols = ['timeidx','price_pct','vwp_pct','spread','tick_spread','ref_ind_0','ref_ind_1','ask_weight_14',
                   'ask_weight_13','ask_weight_12','ask_weight_11','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
                   'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
                   'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
                   'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','bid_weight_11','bid_weight_12','bid_weight_13',
                   'bid_weight_14','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','turnover',
                   'cls_2', 'cls_5', 'cls_18']

class HFDataset(Dataset):

    def __init__(self,
                 local_ids,
                 shard_dict,
                 batch_size,
                 seq_len = 50,
                 **kwargs):
        super().__init__(**kwargs)
        self.shard_dict = shard_dict
        self.seq_len = seq_len
        self._x = []
        self._y = []

        self._load_array(local_ids, batch_size, seq_len)

    def _load_array(self, local_ids, batch_size, seq_len):
        # load from redis
        local_ids.sort()
        rs = ut.redis_connection(db = 0)
        temp_data = []
        temp_len = 0
        for idx in local_ids:
            key = self.shard_dict[idx]
            data = ut.read_data_from_redis(rs, key = key)
            # if len(df) > 6000:
            # df = df.query("tag == 1")
            data = torch.from_numpy(data.astype(np.float32))
            data = data.unfold(dimension=0, size=seq_len, step=seq_len).transpose(1,2)
            temp_data.append(data)
            temp_len += data.size()[0]
            if temp_len >= 500:
                temp_data = torch.cat(temp_data)
                self._x.append(temp_data[:, :, :-3 ])
                self._y.append(temp_data[:, :, -3: ])
                temp_len = 0
                temp_data = []

        if temp_len > 0:
            self._x.append(temp_data[:, :, :-3])
            self._y.append(temp_data[:, :, -3:])

        rs.close()
        return


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self._x[idx], self._y[idx]

    def __len__(self):
        return len(self._x)


class HFDatasetVal(Dataset):

    def __init__(self,
                 local_ids,
                 shard_dict,
                 batch_size,
                 seq_len = 50,
                 **kwargs):
        super().__init__(**kwargs)
        self.shard_dict = shard_dict
        self.seq_len = seq_len
        self._x = []
        self._y = []

        self._load_array(local_ids, batch_size, seq_len)

    def _load_array(self, local_ids, batch_size, seq_len):
        # load from redis
        local_ids.sort()
        rs = ut.redis_connection(db = 0)
        for idx in local_ids:
            key = self.shard_dict[idx]
            data = ut.read_data_from_redis(rs, key = key)
            data = torch.from_numpy(data.astype(np.float32))
            data = data.unfold(dimension=0, size=seq_len, step=seq_len).transpose(1,2)
            # for item in unfold_data:
            self._x.append(data[:, :, :-3 ])
            self._y.append(data[:, :, -3: ])

        rs.close()
        return


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self._x[idx], self._y[idx]

    def __len__(self):
        return len(self._x)

