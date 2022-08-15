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

class HFDatasetTST(Dataset):

    def __init__(self,
                 array,
                 LEN_SEQ = 50,
                 time_step = 1,
                 batch_size = 3000):
        super().__init__()
        self.seq_len = LEN_SEQ
        self._x = []
        self._y = []

        self._load_array(array, LEN_SEQ, time_step, batch_size)

    def _load_array(self, array, seq_len, time_step, batch_size):
        # load from redis
        array = torch.Tensor(array)
        # array = F.pad(array.transpose(0, 1), (seq_len - 1, 0), 'constant').transpose(0, 1)
        unfold_array = array.unfold(dimension=0, size=seq_len, step=seq_len).transpose(1,2)
        last_idx = len(array) - (len(unfold_array) - 1) * seq_len
        self._x.append(unfold_array[: -1, :, : -3])
        self._y.append(unfold_array[: -1, :, -3:])
        self._x.append(array[-last_idx:, :-3].unsqueeze(0))
        self._y.append(array[-last_idx:, -3:].unsqueeze(0))
        return


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self._x[idx], self._y[idx]

    def __len__(self):
        return len(self._x)


class HFDataset(Dataset):

    def __init__(self,
                 array,
                 LEN_SEQ = 50,
                 time_step = 1,
                 batch_size = 3000):
        super().__init__()
        self.seq_len = LEN_SEQ
        self._x = []
        self._y = []

        self._load_array(array, LEN_SEQ, time_step, batch_size)

    def _load_array(self, array, seq_len, time_step, batch_size):
        # load from redis
        array = torch.Tensor(array)
        array = F.pad(array.transpose(0, 1), (seq_len - 1, 0), 'constant').transpose(0, 1)
        unfold_array = array.unfold(dimension=0, size=seq_len, step=time_step).transpose(1,2)
        batch_cnt = math.ceil(unfold_array.size()[0] / batch_size)
        for i in range(1, batch_cnt + 1):
            self._x.append(unfold_array[(i - 1) * batch_size: i * batch_size, :, : -4])
            self._y.append(unfold_array[(i - 1) * batch_size: i * batch_size, -1, -4:].squeeze(1))
        return


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self._x[idx], self._y[idx]

    def __len__(self):
        return len(self._x)


class HFDatasetCls(Dataset):

    def __init__(self,
                 array,
                 LEN_SEQ = 50,
                 time_step = 1,
                 batch_size = 3000):
        super().__init__()
        self.seq_len = LEN_SEQ
        self._x = []
        self._y = []

        self._load_array(array, LEN_SEQ, time_step, batch_size)

    def _load_array(self, array, seq_len, time_step, batch_size):
        # load from redis
        array = torch.Tensor(array)
        array = F.pad(array.transpose(0, 1), (seq_len - 1, 0), 'constant').transpose(0, 1)
        unfold_array = array.unfold(dimension=0, size=seq_len, step=time_step).transpose(1,2)
        batch_cnt = math.ceil(unfold_array.size()[0] / batch_size)
        for i in range(1, batch_cnt + 1):
            self._x.append(unfold_array[(i - 1) * batch_size: i * batch_size, :, : -4])
            self._y.append(unfold_array[(i - 1) * batch_size: i * batch_size, -1, -4:].squeeze(1))
        return


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self._x[idx], self._y[idx]

    def __len__(self):
        return len(self._x)

