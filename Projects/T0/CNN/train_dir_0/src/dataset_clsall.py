import json
from typing import Optional
from collections import Counter

import numpy as np
from math import ceil
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import train_dir_0.utilities as ut
import gc



y_cols = ['cls_2', 'cls_5', 'cls_18', 'subcls_2', 'subcls_5', 'subcls_18', 'p_2','p_5','p_18']

factor_ret_cols = ['timeidx','price_pct','vwp_pct','spread','tick_spread','ref_ind_0','ref_ind_1','ask_weight_14',
                   'ask_weight_13','ask_weight_12','ask_weight_11','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
                   'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
                   'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
                   'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','bid_weight_11','bid_weight_12','bid_weight_13',
                   'bid_weight_14','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','turnover',
                   'cls_2', 'cls_5', 'cls_18', 'subcls_2', 'subcls_5', 'subcls_18', 'p_2','p_5','p_18']


class HFDataset5cls(Dataset):

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
        self._class_count = [torch.zeros(5, ) for i in range(4)]
        self.assist_vector = torch.arange(5)
        self.deducted_vector = torch.ones((5,))
        self._load_array(local_ids, batch_size, seq_len, time_step)

    def _load_array(self, local_ids, batch_size, seq_len, time_step):
        # load from redis
        local_ids.sort()
        rs = ut.redis_connection(db=0)

        for idx in local_ids:
            key = self.shard_dict[idx]
            data = ut.read_data_from_redis(rs, key=key).astype(np.float32)
            data = torch.from_numpy(data)
            unfold_data = data.unfold(dimension=0, size=seq_len, step=time_step).transpose(1, 2)
            batch_num = math.ceil(len(unfold_data) / batch_size)
            for i in range(4):
                temp = torch.cat((data[:, -(8 - i)], self.assist_vector))
                temp = torch.unique(temp, return_counts=True)[1] - self.deducted_vector

                self._class_count[i] += temp

            for i in range(1, batch_num + 1):
                if i == batch_num:
                    break

                _x = unfold_data[(i - 1) * batch_size: i * batch_size, :, 2: -8]

                if _x[:, -1, :].sum() == 0:
                    continue

                _y = unfold_data[(i - 1) * batch_size: i * batch_size, -1, -8:-4].long()

                self._x.append(_x)
                self._y.append(_y)

        rs.close()
        return

    def get_labels_weights(self):

        res = []
        for i in range(4):
            # if i == 0 or i == 3:
            #     self._class_count[i] += torch.Tensor([0, 1e20, 0, 1e20, 0])
            res.append((1. / self._class_count[i]).unsqueeze(0))
        res = torch.cat(res)

        return res

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self._x[idx], self._y[idx]

    def __len__(self):
        return len(self._x)