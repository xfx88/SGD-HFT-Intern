import json
from typing import Optional
from collections import Counter

import numpy as np
from math import ceil
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch.nn.functional as F
import torch.nn as nn
import torch
from functools import partial
import math
import train_dir_0.utilities as ut
import gc



factor_ret_cols = ['timeidx','price','vwp','spread','tick_spread','ref_ind_0','ref_ind_1','ask_weight_14',
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
                 seq_len=50,
                 **kwargs):
        super().__init__(**kwargs)
        self.shard_dict = shard_dict
        self.seq_len = seq_len
        self._x = []
        self._y = []

        self._load_array(local_ids, seq_len)

    def _load_array(self, local_ids, seq_len):
        # load from redis
        local_ids.sort()
        rs = ut.redis_connection(db = 1)

        for idx in local_ids:
            key = self.shard_dict[idx]
            data = ut.read_data_from_redis(rs, key = key).astype(np.float32)
            data = torch.from_numpy(data)
            unfold_data = data.unfold(dimension = 0, step = seq_len, size = seq_len).transpose(1, 2)
            self._x.append(unfold_data[:, :, :-4])
            self._y.append(unfold_data[:, :, -4:-1])

        rs.close()
        return


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self._x[idx], self._y[idx]

    def __len__(self):
        return len(self._x)
