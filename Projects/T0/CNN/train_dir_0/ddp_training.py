
import sys
sys.path.append("/home/yby/SGD-HFT-Intern/Projects/T0/CNN")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.multiprocessing as mp
from collections import defaultdict
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import fast_soft_sort.pytorch_ops as torchsort

import utilities as ut

import math
import random
import pickle
import os
import gc
import src.logger as logger
from src.dataset import HFDataset
from utilities import *

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9"

DB_ID = 0

Logger = logger.getLogger()
Logger.setLevel("INFO")


GLOBAL_SEED = 2098
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

BATCH_SIZE = 5000
WARMUP = 4000
RESUME = None

INPUT_SIZE = 43
OUTPUT_SIZE = 3
SEQ_LEN = 64
TIMESTEP = 5

EPOCHS = 80

validation_dates = ['20211020','20211021','20211022','20211025','20211026','20211027','20211028','20211029']

def corrcoef(target, pred):
    pred_n = pred - pred.mean() + torch.rand(pred.shape).cuda() * 1e-12
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return (pred_n * target_n).sum()

def spearman(
    target,
    pred,
    regularization="kl",
    regularization_strength=1.0,
    weight = (0.2, 0.3, 0.5)):

    corr = 0
    for i in range(pred.shape[-1]):
        pred_i = pred[:, i : i + 1].T
        target_i = target[:, i : i + 1].T
        pred_i = torchsort.soft_rank(
                 pred_i,
                 regularization=regularization,
                 regularization_strength=regularization_strength,)

        corr += weight[i] * corrcoef(target_i, pred_i / pred_i.shape[-1])

    return 1 - corr

class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()
        self.l1loss = nn.SmoothL1Loss()

    def forward(self, pred, target):
        spearman_loss = spearman(pred = pred, target = target)
        l1 = self.l1loss(pred, target)
        return 1e6 * spearman_loss * l1

# class ConvLstmNet(nn.Module):
#     def __init__(self, in_features, seq_len, out_features):
#         super(ConvLstmNet, self).__init__()
#         # set size
#         self.in_features = in_features
#         self.seq_len     = seq_len
#         self.out_features = out_features
#
#         self.dropout = nn.Dropout(0.3)
#
#         self.BN = nn.BatchNorm1d(num_features=in_features)
#
#         self.conv_1 = nn.Conv1d(in_channels=in_features,
#                                 out_channels=32,
#                                 kernel_size=(3,),
#                                 padding=(1,),
#                                 bias=False)
#         self.conv_2 = nn.Conv1d(in_channels=32,
#                                 out_channels=64,
#                                 kernel_size=(3,),
#                                 padding=(0, ),
#                                 bias=False)
#         self.conv_3 = nn.Conv1d(in_channels=64,
#                                     out_channels=128,
#                                     kernel_size=(3,),
#                                     padding=(1, ),
#                                     bias=False)
#         self.conv_4 = nn.Conv1d(in_channels=128,
#                                 out_channels=256,
#                                 kernel_size=(3,),
#                                 padding=(1,),
#                                 bias=False)
#         self.conv_5 = nn.Conv1d(in_channels=256,
#                                 out_channels=512,
#                                 kernel_size=(3,),
#                                 padding=(1,),
#                                 bias=False)
#         self.conv_6 = nn.Conv1d(in_channels=512,
#                                 out_channels=128,
#                                 kernel_size=(3,),
#                                 padding=(1,),
#                                 bias=False)
#         # self.Relu0 = nn.ReLU()
#         self.rnn = nn.GRU(128, 64, num_layers=1)
#
#         self.Relu1 = nn.ReLU()
#         self.downsample = nn.Conv1d(in_channels=64,
#                                     out_channels=4,
#                                     kernel_size=(SEQ_LEN,),
#                                     bias=False)
#
#
#
#     def forward(self, x):
#         x = self.BN(x)
#         conv_feat = self.conv_1(x)
#         conv_feat = self.conv_2(conv_feat)
#         conv_feat = self.conv_3(conv_feat)
#
#         conv_feat = self.dropout(conv_feat)
#
#         conv_feat = self.conv_4(conv_feat)
#         conv_feat = self.conv_5(conv_feat)
#         conv_feat = self.conv_6(conv_feat)
#
#         conv_feat = self.dropout(conv_feat)
#
#         conv_feat = conv_feat.permute(2,0,1)
#         rnn_feat,_  = self.rnn(conv_feat)
#
#         rnn_feat = self.Relu1(rnn_feat)
#         y = self.downsample(rnn_feat.permute(1,2,0)).squeeze(-1)
#         return y

class ConvLstmNet(nn.Module):
    def __init__(self, in_features, seq_len, out_features):
        super(ConvLstmNet, self).__init__()
        # set size
        self.in_features = in_features
        self.seq_len     = seq_len
        self.out_features = out_features

        self.dropout = nn.Dropout(0.3)

        self.BN = nn.BatchNorm1d(num_features=in_features)

        self.conv_1 = nn.Conv1d(in_channels=in_features,
                                out_channels=128,
                                kernel_size=(3,),
                                padding=(1,),
                                bias=False)
        self.conv_2 = nn.Conv1d(in_channels=128,
                                out_channels=256,
                                kernel_size=(3,),
                                padding=(1, ),
                                bias=False)
        self.conv_3 = nn.Conv1d(in_channels=256,
                                    out_channels=512,
                                    kernel_size=(3,),
                                    padding=(1, ),
                                    bias=False)
        self.conv_4 = nn.Conv1d(in_channels=512,
                                out_channels=256,
                                kernel_size=(3,),
                                padding=(1,),
                                bias=False)
        self.conv_5 = nn.Conv1d(in_channels=256,
                                out_channels=128,
                                kernel_size=(3,),
                                padding=(1,),
                                bias=False)
        self.conv_6 = nn.Conv1d(in_channels=128,
                                out_channels=64,
                                kernel_size=(3,),
                                padding=(1,),
                                bias=False)

        self.relu0 = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.rnn = nn.GRU(64, 8, num_layers=1)

        self.downsample = nn.Conv1d(in_channels=8,
                                    out_channels=OUTPUT_SIZE,
                                    kernel_size=(SEQ_LEN,),
                                    bias=False)

    def forward(self, x):
        x = self.BN(x)
        conv_feat = self.conv_1(x)
        conv_feat = self.conv_2(conv_feat)
        conv_feat = self.conv_3(conv_feat)
        conv_feat = self.relu0(conv_feat)

        conv_feat = self.conv_4(conv_feat)
        conv_feat = self.conv_5(conv_feat)
        conv_feat = self.conv_6(conv_feat)
        conv_feat = self.relu1(conv_feat)

        conv_feat = conv_feat.permute(2,0,1)

        rnn_feat,_  = self.rnn(conv_feat)
        rnn_feat = self.dropout(rnn_feat)

        y = self.downsample(rnn_feat.permute(1,2,0)).squeeze(-1)

        return y


class TickMSE(nn.Module):
    def __init__(self, scale_factor, local_rank, reduction = 'mean'):
        super().__init__()
        self.scale_factor = scale_factor
        # self.base_loss = nn.MSELoss(reduction=reduction)
        self.base_loss = nn.SmoothL1Loss(reduction=reduction)

    def forward(self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor) -> torch.Tensor:
        loss = self.scale_factor * (self.base_loss(y_pred, y_true))

        return loss

class Optimiser:

    def __init__(self, model, optimiser=None, scale_factor=1.0, warmup_steps=2000, beta1=0.9, beta2=0.98, epsilon=1e-9):

        if optimiser is not None: self.optimiser = optimiser
        else: self.optimiser = torch.optim.AdamW(model.parameters(), lr=0, betas=(beta1, beta2), eps=epsilon)

        self.scale_factor = scale_factor
        self.warmup_steps = math.pow(warmup_steps, -1.5)
        self.current_step = 0
        self.inv_sqrt_d_input = math.pow(model.module.in_features, -0.5)

        self.lrate = lambda step: self.inv_sqrt_d_input * min(math.pow(step, -0.5), step * self.warmup_steps)
        self.rate = None

    def step(self):
        self.current_step += 1
        self.rate = self.scale_factor * self.lrate(self.current_step)
        for i in self.optimiser.param_groups: i['lr'] = self.rate
        self.optimiser.step()


def shard_keys(start_date, end_date, seq_len = 50, time_step = 1, db = DB_ID):
    shard_dict_whole = dict()
    rs = ut.redis_connection(db=db)
    all_redis_keys = rs.keys()
    keys_to_shard = [x.decode(encoding = 'utf-8') for x in all_redis_keys
                    if ((len(x.decode(encoding = 'utf-8').split('_')) == 3)
                    and (x.decode(encoding = 'utf-8').split('_')[2] <= end_date[4:6])
                    and (x.decode(encoding = 'utf-8').split('_')[2] >= start_date[4:6])
                    and (x.decode(encoding = 'utf-8').split('_')[0] == 'numpy'))]
    cnt = 0 # 记录所有序列的长度
    keys_to_shard.sort()
    for key in keys_to_shard:
        shard_dict_whole[cnt] = key
        cnt += 1
    rs.close()

    return shard_dict_whole, cnt


def distribute_to_child(start_date, end_date, seq_len = 50, time_step = 1, db = DB_ID, world_size = 1):
    shard_dict, key_num = shard_keys(start_date, end_date, seq_len = seq_len, time_step = time_step, db = db)
    idx_list = list(range(key_num))
    # random.seed(GLOBAL_SEED)
    len_part = math.ceil(key_num / world_size)
    random.Random(GLOBAL_SEED).shuffle(idx_list)
    world_dict = {i: [idx_list[j] for j in idx_list[i * len_part: (i + 1) * len_part]]
                  for i in range(world_size)}

    # with open(f'shards/world_dict_bs{BATCH_SIZE}_seed{GLOBAL_SEED}_warmup{WARMUP}.pkl', 'w') as f:
    #     pickle.dump(world_dict, f)
    #     f.close()
    #
    # with open(f'shards/shard_dict_{start_date}_{end_date}.pkl') as f:
    #     pickle.dump(shard_dict, f)
    #     f.close()

    return world_dict, shard_dict


def train(local_rank, world_size, world_dict, shard_dict, validation = False, Resume = None):
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    model_path = 'model/CNN/param_2/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if Resume:
        model_name = f'CNN_epoch_2_bs8000_sl64_ts5.pth.tar'
        model_data = torch.load(os.path.join(model_path, model_name), map_location='cpu')
        model_data['state_dict'] = ddpModel_to_normal(model_data['state_dict'])

    LOGGER = logger.getLogger()
    LOGGER.setLevel('INFO') if local_rank == 0 else LOGGER.setLevel('WARNING')
    prior_epochs = 0 if not Resume else model_data['epoch'] + 1

    net = ConvLstmNet(in_features=INPUT_SIZE,
                      seq_len=SEQ_LEN,
                      out_features=OUTPUT_SIZE).to(local_rank)


    if prior_epochs != 0:
        LOGGER.info(f"Prior epoch: {prior_epochs}, training resumes.")
        net.load_state_dict(model_data['state_dict'])
    else:
        LOGGER.info(f"No prior epoch, training start.")

    ddp_net = DDP(net, device_ids=[local_rank], output_device=local_rank)
    loss_function = myLoss().to(local_rank)
    # loss_function = TickMSE(scale_factor=1e6, local_rank = local_rank).to(local_rank)
    opt = Optimiser(ddp_net, scale_factor = 0.08, warmup_steps=WARMUP)

    if prior_epochs != 0:
        opt.optimiser.load_state_dict(model_data['optimizer_dict'])
        opt.current_step = model_data['current_step']

    ddp_net.train()
    local_rank_id = world_dict[local_rank]
    local_rank_id.sort()
    len_train = int(len(local_rank_id) * 0.8)
    train_ids = local_rank_id[:len_train]
    val_ids = local_rank_id[len_train:]
    train_dataset = HFDataset(local_ids = train_ids,
                              shard_dict = shard_dict,
                              batch_size = BATCH_SIZE,
                              seq_len=SEQ_LEN,
                              time_step = TIMESTEP)

    val_dataset = HFDataset(local_ids=val_ids,
                            shard_dict=shard_dict,
                            batch_size=BATCH_SIZE,
                            seq_len=SEQ_LEN,
                            time_step=TIMESTEP)

    LOGGER.info(f'local rank {local_rank}: Dataset is prepared.')
    LOGGER.info(f'local rank {local_rank}: GPU calculating.')
    datalen = len(train_dataset)
    INPUT_SHAPE = (-1, INPUT_SIZE, SEQ_LEN, 1)
    for epoch_idx in range(prior_epochs, EPOCHS):
        total_loss = 0.0
        if epoch_idx != 0 and epoch_idx % 7 == 0: opt.scale_factor /= 2

        for batch_idx, (x, y) in enumerate(train_dataset):
            opt.optimiser.zero_grad()
            netout = net(x.permute(0,2,1).to(local_rank))

            loss = loss_function(netout, y.to(local_rank))
            total_loss += loss.item()

            loss.backward()
            opt.step()

            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                LOGGER.info(f'Epoch {epoch_idx}: {batch_idx + 1} / {datalen}), loss {loss.item()}.')

        if local_rank == 0:
            dist.barrier()
            if validation:
                val_loss = validate(net, val_dataset, loss_function, local_rank)
                LOGGER.info(f'Epoch {epoch_idx} validation loss: {val_loss}.')
            torch.save({
                'epoch': epoch_idx,
                'train_loss': total_loss / datalen,
                'validation_loss': val_loss,
                'state_dict': ddp_net.state_dict(),
                'current_step': opt.current_step,
                'optimizer_dict': opt.optimiser.state_dict(),
            }, os.path.join(model_path, f'CNN_epoch_{epoch_idx}_bs{BATCH_SIZE}_sl{SEQ_LEN}_ts{TIMESTEP}.pth.tar'))
            LOGGER.info(f'Epoch {epoch_idx} finished.')

        else:
            dist.barrier()

        del loss
        gc.collect()
        torch.cuda.empty_cache()

        LOGGER.info(f"Epoch {epoch_idx} is done.\n")

    LOGGER.info("Training has finished.")

    return

def validate(net, validation_dataset, loss_function, local_rank):
    INPUT_SHAPE = (-1, INPUT_SIZE, SEQ_LEN, 1)
    running_loss = 0
    with torch.no_grad():
        for x, y in validation_dataset:
            netout = net(x.permute(0,2,1).to(local_rank))

            loss = loss_function(netout, y.to(local_rank))
            running_loss += loss.item()
    return running_loss / len(validation_dataset)

def main_train(LOGGER):

    train_start_date = '20210701'
    train_end_date = '20210930'
    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    LOGGER.info('Splitting indices for distributed training.')
    world_dict, shard_dict = distribute_to_child(start_date = train_start_date ,
                                                end_date = train_end_date,
                                                seq_len = SEQ_LEN,
                                                time_step = TIMESTEP,
                                                db = DB_ID,
                                                world_size = world_size)

    # world_size = 1
    mp.spawn(train,
             args=(world_size, world_dict, shard_dict, True, None, ),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    main_train(Logger)