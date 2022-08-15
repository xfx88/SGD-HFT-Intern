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



factor_ret_cols = ['timeidx', 'price', 'vwp', 'spread', 'tick_spread', 'ref_ind_0', 'ref_ind_1',
                   'ask_weight_14', 'ask_weight_13', 'ask_weight_12', 'ask_weight_11',
                   'ask_weight_10', 'ask_weight_9', 'ask_weight_8', 'ask_weight_7',
                   'ask_weight_6', 'ask_weight_5', 'ask_weight_4', 'ask_weight_3',
                   'ask_weight_2', 'ask_weight_1', 'ask_weight_0', 'bid_weight_0',
                   'bid_weight_1', 'bid_weight_2', 'bid_weight_3', 'bid_weight_4',
                   'bid_weight_5', 'bid_weight_6', 'bid_weight_7', 'bid_weight_8',
                   'bid_weight_9', 'bid_weight_10', 'bid_weight_11', 'bid_weight_12',
                   'bid_weight_13', 'bid_weight_14', 'ask_dec', 'bid_dec', 'ask_inc',
                   'bid_inc', 'ask_inc2', 'bid_inc2', '2', '5', '10', '20']



class HFDataset(Dataset):

    def __init__(self,
                 data_array,
                 tick_num = 200,
                 LEN_SEQ=100,
                 # batch_num = 2,
                 # step = 10,
                 normalize: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)

        self._normalize = normalize
        # self.batch_num = batch_num
        # self.step = step
        self.batch_num = int(4800 / LEN_SEQ)
        self.feature_size = data_array[0].shape[-1]
        self._load_array(data_array, tick_num, LEN_SEQ)

    def _load_array(self, arr, tick_num, LEN_SEQ):
        # Load the array obtained from Redis
        maxlen = 4800
        self.data = []

        arr = torch.Tensor(arr)
        reshaped_arrs = [F.normalize(arr[:, i * LEN_SEQ : (i + 1) * LEN_SEQ, :], dim = 0) for i in range(self.batch_num)]
        self.data.extend(reshaped_arrs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx][..., :-5], self.data[idx][..., -4:]

    def __len__(self):
        return len(self.data)