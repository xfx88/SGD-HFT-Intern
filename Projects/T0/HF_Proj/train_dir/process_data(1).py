import pandas as pd
import torch
import numpy as np
import os
import rqdatac as rq
import torch.nn.functional as F
import utilities as ut
import time
import torch.multiprocessing as mp
from torch import distributed as dist
from joblib import Parallel,delayed
from scipy.stats import spearmanr


tick_nums = 200
minibatch_size = 150

factor_numbers = 43
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,5,6,7,8,9"


def get_samples(key, tick_nums):
    rs = ut.redis_connection()
    data = ut.read_data_from_redis(rs, key)[factor_ret_cols].fillna(0)
    data = data.values.astype(np.float32)
    if len(data) < tick_nums:
        return
    zero_tensor = torch.zeros((tick_nums - 1, len(factor_ret_cols)))
    data_t = torch.cat((zero_tensor,torch.tensor(data)))
    data2 = data_t.unfold(0, 200, 1).unsqueeze(dim = 3)
    rs.close()
    # print("处理完成 ", key, os.getpid())
    return data2


def gen_processed_data_to_redis(file_path, df_full_time):
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


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def train(rank, train_redis_keys, cirterion, epoch, cnn_model, optimizer):
    # 获取数据
    batch_size = 5
    ranks_number = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    rank_batch = list(split(train_redis_keys, ranks_number))

    for i in range(0, len(rank_batch[rank]), batch_size):
        if i + batch_size > len(rank_batch[rank]):
            batch_keys = rank_batch[rank][i:]
        else:
            batch_keys = rank_batch[rank][i:i + batch_size]
        batch_keys.sort(key=lambda x: str(x).split('_')[-2])
        process = []
        for key in batch_keys:
            res = get_samples(key, tick_nums)
            process.append(res)
        batch_data = torch.cat(process)
        inputs = batch_data[:, :factor_numbers, :, :]
        labels = batch_data[:, factor_numbers, -1, :]
        optimizer.zero_grad()
        y_pred = cnn_model(inputs).to("cpu").type(torch.DoubleTensor)
        labels = labels.type(torch.DoubleTensor)
        loss = cirterion(y_pred, labels)
        loss.backward()
        optimizer.step()
        print('rank: {}, epoch: {}, batch: {}-{}, running_loss: {}'.format(
            rank, epoch, i, i+batch_size, loss.item())
        )
    if rank == 0:
        torch.save({
            'epoch': epoch,
            'model': cnn_model,
            'optimizer': optimizer,
            'state_dict': cnn_model.state_dict(),
            'optimizer_dict': optimizer.state_dict(),
        }, os.path.join("output", f"cnn_model_epoch_{epoch}_ddp.pth.tar"))
    return cnn_model, optimizer


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


def model_eval(rank, world_size, test_redis_keys, model_name):
    """
    模型测试
    """
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print("model_eval rank:", rank)
    model_data = torch.load(model_name)
    model = model_data["model"]
    model_state_dict = model_data["state_dict"]
    model.load_state_dict(model_state_dict)
    model.eval()
    batch_size = 5
    ics = []

    ranks_number = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    rank_batch = list(split(test_redis_keys, ranks_number))

    for i in range(0, len(rank_batch[rank]), batch_size):
        if i + batch_size > len(rank_batch[rank]):
            batch_keys = rank_batch[rank][i:]
        else:
            batch_keys = rank_batch[rank][i:i + batch_size]
        batch_keys.sort(key=lambda x: str(x).split('_')[-2])
        process = []
        for key in batch_keys:
            res = get_samples(key, tick_nums)
            process.append(res)
        batch_data = torch.cat(process)

        inputs = batch_data[:, :factor_numbers, :, :]
        labels = batch_data[:, factor_numbers, -1, :]
        y_pred = model(inputs).to("cpu").type(torch.DoubleTensor)
        labels = labels.type(torch.DoubleTensor)
        spm = spearmanr(y_pred.detach().numpy(), labels.detach().numpy())
        ics.append(spm)
        torch.cuda.empty_cache()
        print('batch: {}, spm: {}'.format(i, spm))


def run_train(rank, world_size, train_redis_keys):
    print("rank, world_size:", rank, world_size)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    # print(local_rank)

    cnn_model = CNNModel().cuda(device=rank)
    cnn_model = torch.nn.parallel.DistributedDataParallel(cnn_model, device_ids=[rank])
    cirterion = torch.nn.MSELoss(size_average=True)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-5)
    for i in range(0, 100):
        a = time.time()
        train(rank, train_redis_keys, cirterion, i, cnn_model, optimizer)
        print("calculate one epoch time:", time.time() - a)

def main():
    world_size = 6
    user = 18123610100
    pw = 123456
    rq.init(user, pw)

    path = '/sgd-data/t0_data/500factor/500factors'
    allpath, allname = ut.getallfile(path)
    # filepath = allpath[0]
    # data = pd.read_csv(filepath)
    df_full_time = ut.gen_df_full_time()

    train_start_date = 20210701
    train_end_date = 20210930
    test_start_date = 20211001
    test_end_date = 20211031
    # trade_days = rq.get_trading_dates(start_date=train_start_date, end_date=test_end_date)
    # trade_days_str = [(str(x).replace('-', '')) for x in trade_days]

    train_file_path = [x for x in allpath if int(x.split('/')[-1][:-4]) <= train_end_date]
    test_file_path = [x for x in allpath if int(x.split('/')[-1][:-4]) <= test_end_date and
                       int(x.split('/')[-1][:-4]) >= test_start_date]

    rs = ut.redis_connection()
    redis_keys = list(rs.keys())
    cnn_redis_keys = [x for x in redis_keys if 'CNN' in str(x)]
    train_redis_keys = [x for x in cnn_redis_keys if (int(str(x).split('_')[1]) <= train_end_date)
                        and (int(str(x).split('_')[1]) >= train_start_date)]
    test_redis_keys = [x for x in cnn_redis_keys if (int(str(x).split('_')[1]) <= test_end_date)
                        and (int(str(x).split('_')[1]) >= test_start_date)]

    if len(test_redis_keys) == 0:
        Parallel(n_jobs=100, verbose=2, timeout=10000)(delayed(gen_processed_data_to_redis)(file_path, df_full_time)
                                                       for file_path in test_file_path)
    if len(cnn_redis_keys) == 0:
        Parallel(n_jobs=100, verbose=2, timeout=10000)(delayed(gen_processed_data_to_redis)(file_path, df_full_time)
                                                       for file_path in train_file_path)
    mp.spawn(run_train,
             args=(world_size, train_redis_keys,),
             nprocs=world_size,
             join=True)
    # mp.spawn(model_eval,
    #          args=(world_size, test_redis_keys, "/home/zjr/cnn_test/output/cnn_model_epoch_82_ddp.pth.tar",),
    #          nprocs=world_size,
    #          join=True)

if __name__ == "__main__":
    main()
