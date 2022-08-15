# -*- coding; utf-8 -*-
"""
Project: main.py
File: dataset_generator.PY
Time: 10:04
Date: 2022/6/20
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""
import math
import random

import torch

import utilities as ut
import numpy as np



class DatasetAll:

    def __init__(self, local_ids,
                       shard_dict,
                       batch_size,
                       seq_len = 50,
                       time_step = 2,):
        super(DatasetAll, self).__init__()
        self.shard_dict = shard_dict
        self.seq_len = seq_len
        local_ids.sort()
        self.local_ids = local_ids
        self.batch_size = batch_size
        self.time_step = time_step

        self._class_count = [torch.zeros(5, ) for i in range(4)]
        self.assist_vector = torch.arange(5)
        self.deducted_vector = torch.ones((5,))

        self.data_generator = self._fetch_data()


    def _fetch_data(self, ):
        rs = ut.redis_connection(db=0)

        for idx in self.local_ids:

            key = self.shard_dict[idx]
            data = ut.read_data_from_redis(rs, key=key).astype(np.float32)
            data = torch.from_numpy(data)
            unfold_data = data.unfold(dimension=0, size=self.seq_len, step=self.time_step).transpose(1, 2)
            batch_num = math.ceil(len(unfold_data) / self.batch_size)
            # for i in range(4):
            #     temp = torch.cat((data[:, -(8 - i)], self.assist_vector))
            #     temp = torch.unique(temp, return_counts=True)[1] - self.deducted_vector
            #
            #     self._class_count[i] += temp

            for i in range(1, batch_num + 1):
                if i == batch_num:
                    break

                _x = unfold_data[(i - 1) * self.batch_size: i * self.batch_size, :, : -8]

                if _x[:, -1, :].sum() == 0:
                    continue

                _y = unfold_data[(i - 1) * self.batch_size: i * self.batch_size, -1, -8:-5].long()

                yield _x, _y

        rs.close()


    def __getitem__(self, item):

        return next(self.data_generator)



class DatasetGenerator:

    def __init__(self,
                 start_date,
                 end_date,
                 local_rank,
                 world_size,
                 batch_size,
                 seq_len = 64,
                 time_step = 2,
                 train_partition = 0.8):

        self.start_date = start_date
        self.end_date = end_date
        self.local_rank = local_rank
        self.world_size = world_size
        self.train_partition = train_partition
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.time_step = time_step
        self._generate_keys()


    def update(self, epoch_idx):
        idx_train = list(range(self.train_cnt))
        idx_valid = list(range(self.valid_cnt))

        len_train = math.ceil(self.train_cnt / self.world_size)
        len_valid = math.ceil(self.valid_cnt / self.world_size)

        random.Random(19491).shuffle(idx_train)
        # random.Random(epoch_idx).shuffle(idx_train)
        random.Random(19491).shuffle(idx_valid)

        local_train_id = [idx_train[j] for j in idx_train[self.local_rank * len_train: (self.local_rank + 1) * len_train]]
        local_valid_id = [idx_valid[j] for j in idx_valid[self.local_rank * len_valid: (self.local_rank + 1) * len_valid]]

        return DatasetAll(local_train_id, self.shard_dict_train, self.batch_size, self. seq_len, self.time_step), \
               DatasetAll(local_valid_id, self.shard_dict_valid, self.batch_size, self. seq_len, self.time_step)

    def _generate_keys(self):
        shard_dict_train = dict()
        shard_dict_valid = dict()
        rs = ut.redis_connection()
        all_redis_keys = rs.keys()
        keys_to_shard = [x.decode(encoding='utf-8') for x in all_redis_keys
                         if ((len(x.decode(encoding='utf-8').split('_')) == 3)
                             and (x.decode(encoding='utf-8').split('_')[1] <= self.end_date[4:6])
                             and (x.decode(encoding='utf-8').split('_')[1] >= self.start_date[4:6])
                             and (x.decode(encoding='utf-8').split('_')[0] == 'manulabels'))]

        keys_to_shard.sort()

        len_train = math.ceil(len(keys_to_shard) * self.train_partition)

        keys_train = keys_to_shard[:len_train]
        keys_valid = keys_to_shard[len_train:]

        train_cnt = 0
        valid_cnt = 0  # 记录所有序列的长度

        for key in keys_train:
            shard_dict_train[train_cnt] = key
            train_cnt += 1
        rs.close()

        for key in keys_valid:
            shard_dict_valid[valid_cnt] = key
            valid_cnt += 1
        rs.close()

        self.keys_train = keys_train
        self.keys_valid = keys_valid
        self.train_cnt = train_cnt
        self.valid_cnt = valid_cnt
        self.shard_dict_train = shard_dict_train
        self.shard_dict_valid = shard_dict_valid