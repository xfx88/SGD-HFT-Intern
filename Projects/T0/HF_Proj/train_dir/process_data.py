import datetime
import pandas as pd
import pympler.asizeof
import torch
import numpy as np
import os
from joblib import Parallel,delayed
import rqdatac as rq
from pympler import asizeof
import torch.nn.functional as F
import utilities as ut
import time
import argparse
from torch import distributed as dist

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


tick_nums = 200
minibatch_size = 150

factor_numbers = 43
# os.environ["RANK"] = "0"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"
# os.environ["MASTER_ADDR"] = "127.0.0.1"
# os.environ["MASTER_PORT"] = "8080"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6"
# os.environ["TP_SOCKET_IFNAME"] = "wg0"
# os.environ["GLOO_SOCKET_IFNAME"] = "wg0"
#
# torch.distributed.init_process_group(backend="nccl")  # 并行训练初始化，建议'nccl'模式

# print(torch.cuda.device_count())  # 打印gpu数量
# print('world_size', torch.distributed.get_world_size())  # 打印当前进程数

# parser = argparse.ArgumentParser(description='Pytorch ...')
# parser.add_argument('--local_rank', default=-1, type=int)
# args = parser.parse_args()
# torch.cuda.set_device(args.local_rank)  # 设置gpu编号为local_rank;此句也可能看出local_rank的值是什么


def get_samples(key, tick_nums):
    rs = ut.redis_connection()
    data = ut.read_data_from_redis(rs, key)[factor_ret_cols].values.astype(np.float32)
    print(np.argwhere(np.isnan(data)))
    if len(data) < tick_nums:
        return
    zero_tensor = torch.zeros((tick_nums - 1, len(factor_ret_cols)))
    data_t = torch.cat((zero_tensor,torch.tensor(data)))
    data2 = data_t.unfold(0, 200, 1).unsqueeze(dim = 3)
    rs.close()
    # print("处理完成 ", key, os.getpid())
    return data2


def gen_processed_data_to_redis(file_path):
    rs = ut.redis_connection()
    data = pd.read_csv(file_path)
    data.columns = col_factors
    code = file_path.split('/')[-2]
    date = file_path.split('/')[-1][:-4]
    data.insert(2, 'code', code)
    data = ut.get_target(data, df_full_time)
    ut.save_data_to_redis(rs, f'CNN_{date}_{code}', data)
    rs.close()
    return


def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    """
    @param tensor: 多gpu运行结果
    @param nprocs: 多少个进程
    @return: 平均值
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

from torch.utils.data import Dataset, DataLoader


class trainset(Dataset):
    def __init__(self, inputs, labels):
        #定义好 image 的路径
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, index):
        fn = self.inputs[index]
        target = self.labels[index]
        return fn, target

    def __len__(self):
        return len(self.inputs)

def train(epoch, cnn_model, optimizer):
    # 获取数据
    running_loss = 0.0
    batch_size = 20
    for i in range(0, len(train_redis_keys), batch_size):
        if i + batch_size > len(train_redis_keys):
            batch_keys = train_redis_keys[i:]
        else:
            batch_keys = train_redis_keys[i:i + batch_size]
        batch_keys.sort(key=lambda x: str(x).split('_')[-2])
        a = time.time()
        process = []
        for key in batch_keys:
            res = get_samples(key, tick_nums)
            process.append(res)
        # print(time.time() - a)
        batch_data = torch.cat(process)
        # print(batch_data.shape)
        inputs = batch_data[:, :factor_numbers, :, :]
        labels = batch_data[:, factor_numbers, -1, :]

        # train_data = trainset(inputs, labels)
        # trainloader = DataLoader(train_data, batch_size=minibatch_size, shuffle=False)
        optimizer.zero_grad()
        # running_loss = 0
        # running_count = 0
        # for j, (minibatch, labels) in enumerate(trainloader):
        #     y_pred = cnn_model(minibatch)
        #     loss = cirterion(y_pred, labels)
        #     running_loss += loss.item()
        #     running_count += 1
        #     loss.backward()
        y_pred = cnn_model(inputs).to("cpu").type(torch.DoubleTensor)
        labels = labels.type(torch.DoubleTensor)
        loss = cirterion(y_pred, labels)
        loss.backward()
        optimizer.step()
        # loss = reduce_mean(torch.Tensor(running_loss), running_count)
        # print("loss: ", loss.item())
        # print(y_pred)
        # print(labels)
        # print(loss.item())
        print('epoch: {}, batch: {}, running_loss: {}'.format(epoch, i, loss.item()))


class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(factor_numbers, 16, kernel_size=(3, 1), padding=(1, 0))
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0))
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0))
        self.conv6 = torch.nn.Conv2d(64, 4, kernel_size=1)
        self.fc = torch.nn.Linear(4 * 200, 1)

    def forward(self, x):
        """
        conv1: torch.Size([150, 16, 200, 1])
        conv2: torch.Size([150, 32, 200, 1])
        conv3: torch.Size([150, 64, 200, 1])
        conv4: torch.Size([150, 128, 200, 1])
        conv5: torch.Size([150, 256, 200, 1])
        conv6: torch.Size([150, 4, 200, 1])
        conv6_: torch.Size([150, 800])
        """
        x = F.relu(self.conv1(x))
        # print("conv1:", x.shape)
        x = F.relu(self.conv2(x))
        # print("conv2:", x.shape)
        x = F.relu(self.conv3(x))
        # print("conv3:", x.shape)
        # x = F.relu(self.conv4(x))
        # print("conv4:", x.shape)
        # x = F.relu(self.conv5(x))
        # print("conv5:", x.shape)
        x = self.conv6(x)
        # print("conv6:", x.shape)
        x = x.view(x.shape[0], -1)
        # print("conv6_:", x.shape)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    user = 18123610100
    pw = 123456
    rq.init(user, pw)

    path = '/sgd-data/t0_data/500factors'
    allpath, allname = ut.getallfile(path)
    filepath = allpath[0]
    data = pd.read_csv(filepath)
    df_full_time = ut.gen_df_full_time()

    train_start_date = 20210701
    train_end_date = 20210930
    test_start_date = 20211001
    test_end_date = 20211031
    trade_days = rq.get_trading_dates(start_date=train_start_date, end_date=test_end_date)
    trade_days_str = [(str(x).replace('-', '')) for x in trade_days]

    train_file_path = [x for x in allpath if int(x.split('/')[-1][:-4]) <= train_end_date]

    rs = ut.redis_connection()
    redis_keys = list(rs.keys())
    cnn_redis_keys = [x for x in redis_keys if 'CNN' in str(x)]
    train_redis_keys = [x for x in cnn_redis_keys if (int(str(x).split('_')[1]) <= train_end_date)
                        and (int(str(x).split('_')[1]) >= train_start_date)]

    if len(cnn_redis_keys) == 0:
        Parallel(n_jobs=100, verbose=2, timeout=10000)(delayed(gen_processed_data_to_redis)(file_path)
                                                       for file_path in train_file_path)
    # print(args.local_rank)
    cnn_model = CNNModel()
    # cnn_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(cnn_model)  # 设置多个gpu的BN同步
    # cnn_model = torch.nn.parallel.DistributedDataParallel(cnn_model,
    #                                                      device_ids=[args.local_rank],
    #                                                      output_device=args.local_rank,
    #                                                      find_unused_parameters=False,
    #                                                      broadcast_buffers=False)
    cnn_model = torch.nn.DataParallel(cnn_model, device_ids=[0, 1, 2])
    cnn_model = cnn_model.cuda()
    cirterion = torch.nn.MSELoss(size_average=True)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-6)

    train(1, cnn_model, optimizer)
