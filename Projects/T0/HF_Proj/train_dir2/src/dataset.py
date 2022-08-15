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
                 tick_num,
                 LEN_SEQ=100,
                 # batch_num = 2,
                 # step = 10,
                 normalize: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.batch_num = int(4800 / LEN_SEQ)
        self.feature_size = data_array[0].shape[-1]
        self.rectify_std = torch.Tensor([0.00075, 0.001, 0.0013, 0.0017])
        self._load_array(data_array, tick_num, LEN_SEQ)

    def _load_array(self, data_array, tick_num, LEN_SEQ):
        # Load the array obtained from Redis
        maxlen = 4800
        self.data = []
        for arr in data_array:
            arr = torch.Tensor(arr)
            if arr.shape[0] > 500:
                arr = arr[:500, ...]
            elif arr.shape[0] < 500:
                arr = F.pad(arr.transpose(0, 2), (500 - arr.shape[0], 0), 'constant').transpose(0,2)
            # arr[..., 0] = arr[..., 0] / 4800
            reshaped_arrs = [arr[:, i * LEN_SEQ : (i + 1) * LEN_SEQ, :] for i in range(self.batch_num)]
            self.data.extend(reshaped_arrs)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx][..., 1: -5], self.data[idx][..., -4:]

    def __len__(self):
        return len(self.data)


class HFDatasetWindow(HFDataset):
    """Torch dataset with windowed time dimension.

    Load dataset from a single npz file.

    Attributes
    ----------
    labels: :py:class:`dict`
        Ordered labels list for R, Z and X.

    Parameters
    ---------
    dataset_x:
        Path to the dataset inputs as npz.
    labels_path:
        Path to the labels, divided in R, Z and X, in json format.
        Default is "labels.json".
    window_size:
        Size of the window to apply on time dimension.
        Default 5.
    padding:
        Padding size to apply on time dimension windowing.
        Default 1.
    """

    def __init__(self,
                 dataset,
                 window_size: int,
                 padding: int,
                 **kwargs):
        """Load dataset from npz."""
        super().__init__(data_array = dataset, tick_num = window_size, **kwargs)

        self._window_dataset(window_size=window_size, padding=padding)

    def _window_dataset(self, window_size=5, padding=1):
        vec_op_padding_x = np.vectorize(op_padding_x)
        vec_op_padding_y = np.vectorize(op_padding_y)
        data_x = list(map(partial(op_padding_x, window_size = window_size, padding = padding), self._x))
        data_y = list(map(partial(op_padding_y, window_size = window_size, padding = padding), self._y))
        data_x = [data[j] for data in data_x for j in range(len(data))]
        data_y = [data[j] for data in data_y for j in range(len(data))]

        self._x = data_x
        self._y = data_y



def op_padding_x(x, window_size, padding):
    _, K, d_input = x.shape

    step = window_size - 2 * padding
    n_step = (K - window_size - 1) // step + 1

    dataset_x = np.empty(
        (1, n_step, window_size, d_input), dtype=np.float32)

    for idx_step, idx in enumerate(range(0, K - window_size, step)):
        dataset_x[:, idx_step, :, :] = x[:, idx:idx + window_size, :]

    return dataset_x

def op_padding_y(y, window_size, padding):
    _, K, d_output = y.shape

    step = window_size - 2 * padding
    n_step = (K - window_size - 1) // step + 1
    dataset_y = np.empty((1, n_step, step, d_output), dtype=np.float32)

    for idx_step, idx in enumerate(range(0, K - window_size, step)):
        dataset_y[:, idx_step, :, :] = y[:, idx + padding:idx + window_size - padding, :]

    return dataset_y
