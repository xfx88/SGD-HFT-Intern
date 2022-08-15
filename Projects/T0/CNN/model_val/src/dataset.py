import json
from typing import Optional
import gc

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
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
                   'bid_inc', 'ask_inc2', 'bid_inc2', '10']



class HFDataset(Dataset):

    def __init__(self,
                 data_array,
                 tick_num,
                 normalize: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.tick_num = tick_num
        self._normalize = normalize
        self._load_array(data_array, tick_num)

    def _load_array(self, data_array, tick_num):
        # Load the array obtained from Redis
        maxlen = 4800
        self._x = []
        self._y = []
        for arr in data_array:
            if not len (arr) >= 4800:
                arr = np.pad(arr, ((0, maxlen - len(arr)), (0, 0)), mode = 'constant')
            # iter_num = int(len(arr) / tick_num + 1)
            # for i in range(iter_num):
            #     arr_to_append = arr[tick_num * i: tick_num *(i + 1)]
            #     if i == iter_num - 1:
            #         padding_array = np.zeros((tick_num - len(arr_to_append), len(factor_ret_cols)))
            #         arr_to_append = np.concatenate((arr_to_append, padding_array))
            self._x.append(arr[:, :-1].reshape(len(arr), -1).astype(np.float32))
            self._y.append(arr[:, -1].reshape(len(arr), -1).astype(np.float32))
        return


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self._x[idx], self._y[idx])

    def __len__(self):
        return len(self._x)

class HFTestDataset(HFDataset):
    def __init__(self, data_array, tick_num):
        super().__init__(tick_num = tick_num, data_array = data_array)

        self._generate_windows()

    def _generate_windows(self):
        self._x_windowed = []
        self._y_windowed = []
        for i in range(len(self._x)):
            _x = torch.Tensor(self._x[i])
            _y = torch.Tensor(self._y[i])
            _x: torch.Tensor = _x.reshape((1, _x.shape[0], -1))
            _y: torch.Tensor = _y.reshape((1, _y.shape[0], -1))
            _x = F.pad(_x.transpose(1, 2), (self.tick_num - 1, 0), 'constant').transpose(1, 2)
            _y = F.pad(_y.transpose(1, 2), (self.tick_num - 1, 0), 'constant').transpose(1, 2)
            _x = _x.unfold(dimension = 1, size = self.tick_num, step = 1).reshape((_x.shape[1] - self.tick_num + 1, self.tick_num, _x.shape[2]))
            _y = _y.unfold(dimension = 1, size = self.tick_num, step = 1).reshape((_y.shape[1] - self.tick_num + 1, self.tick_num, _y.shape[2]))
            for i in range(8):
                self._x_windowed.append(_x[i * 600 : (i + 1) * 600, ...].reshape((-1, _x.shape[-1])))
                self._y_windowed.append(_y[i * 600 : (i + 1) * 600, ...].reshape((-1, _y.shape[-1])))
            del _x, _y
            gc.collect()
        del self._x, self._y
        gc.collect()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self._x_windowed[idx], self._y_windowed[idx])
    
    def __len__(self):
        return len(self._x_windowed)





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

class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples