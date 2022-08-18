"""
5分类模型
"""
import sys
sys.path.append("/home/yby/SGD-HFT-Intern/Projects/T0/CNN")

import numpy as np
import os
import math
import random
import gc

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.multiprocessing as mp
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import weight_norm
from torch.optim import lr_scheduler, SGD


import utilities as ut
import src.logger as logger
from src.dataset_reg import HFDatasetReg
import fast_soft_sort.pytorch_ops as torchsort

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12310"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9"
os.environ["OMP_NUM_THREADS"] = '4'

DB_ID = 0

Logger = logger.getLogger()
Logger.setLevel("INFO")


GLOBAL_SEED = 12308
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

BATCH_SIZE = 10000
WARMUP = 500
RESUME = None

INPUT_SIZE = 44
OUTPUT_SIZE = 3
SEQ_LEN = 64
TIMESTEP = 3

EPOCHS = 150

validation_dates = ['20211020','20211021','20211022','20211025','20211026','20211027','20211028','20211029']

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def corrcoef(target, pred):
    pred_n = pred - pred.mean() + torch.rand(pred.shape) * 1e-18
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return (pred_n * target_n).sum()

def spearman(
    target,
    pred,
    regularization="l2",
    regularization_strength=1.0,
    weight = (1/3, 1/3, 1/3)):

    corr = 0
    for i in range(pred.shape[-1]):
        pred_i = pred[:, i : i + 1].T
        target_i = target[:, i : i + 1].T
        pred_i = torchsort.soft_rank(
                 pred_i,
                 regularization=regularization,
                 regularization_strength=regularization_strength,)

        corr += weight[i] * corrcoef(target_i, pred_i / pred_i.shape[-1])

    return corr


class SelfMadeLoss(nn.Module):
    def __init__(self, local_rank = 0, label_num = 3):
        super(SelfMadeLoss, self).__init__()

        self.label_weight = torch.tensor([1/5, 2/5, 2/5]).to(local_rank)
        self.label_num = range(label_num)
        self.base_losses = [nn.MSELoss(reduction="mean") for i in self.label_num]

    def forward(self, pred, target):

        loss = 0
        for i in self.label_num:
            loss += self.base_losses[i](pred[:, i], target[:, i]) * 1 / len(self.base_losses)

        return loss


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int,
                       out_channels: int,
                       kernel_size: int,
                       dilation_size: int):

        super(CNNBlock, self).__init__()

        _L = SEQ_LEN
        _padding_size = (dilation_size * (kernel_size - 1)) >> 1

        _latent_size = (in_channels + out_channels) >> 1

        self.cnn1 = weight_norm(nn.Conv1d(in_channels=in_channels,
                              out_channels=_latent_size,
                              kernel_size=(kernel_size,),
                              padding=(_padding_size,),
                              dilation=(dilation_size,)))

        self.cnn2 = weight_norm(nn.Conv1d(_latent_size,
                              out_channels=out_channels,
                              kernel_size=(3,),
                              padding=((dilation_size * 2) >> 1,),
                              dilation=(dilation_size,)))

        self.sub_block = nn.Sequential(self.cnn1, nn.ReLU(), nn.Dropout(0.25), self.cnn2, nn.ReLU(), nn.Dropout(0.25))
        self.output_relu = nn.ReLU()
        self.resample = nn.Conv1d(in_channels,
                                  out_channels,
                                  kernel_size=(kernel_size,),
                                  padding=(_padding_size,),
                                  dilation=(dilation_size,)) \
            if in_channels != out_channels else None

        self.init_weight()

    def init_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight.data, nonlinearity='conv1d')
                nn.init.zeros_(layer.bias.data)

    def forward(self, x: torch.Tensor):

        res = self.resample(x) if self.resample else x
        x = self.sub_block(x)
        x = self.output_relu(x + res)

        return x

class ConvLstmNet(nn.Module):
    def __init__(self, in_features, seq_len, out_features):
        super(ConvLstmNet, self).__init__()

        # set size
        self.in_features = in_features
        self.seq_len     = seq_len
        self.out_features = out_features

        _kernel_size = 1

        _layers = []
        _levels = [INPUT_SIZE, 64, 96, 128, 128, 96, 48]
        # _levels = [INPUT_SIZE] * 7
        for i in range(len(_levels)):
            _dilation_size = 1 << max(0, (i - (len(_levels) - 3)))
            _input_size = in_features if i == 0 else _levels[i - 1]
            _output_size = _levels[i]
            _layers.append(CNNBlock(in_channels = _input_size,
                                    out_channels = _output_size,
                                    kernel_size = _kernel_size,
                                    dilation_size = _dilation_size))
        self.network = nn.Sequential(*_layers)
        self.featureExtractor = nn.Linear(_levels[-1] * seq_len, OUTPUT_SIZE)


    def forward(self, x: torch.Tensor):

        x = self.network(x)
        x = self.featureExtractor(x.flatten(1))

        return x

def shard_keys(start_date, end_date, seq_len = 50, time_step = 1, db = DB_ID):
    shard_dict_whole = dict()
    rs = ut.redis_connection(db=db)
    all_redis_keys = rs.keys()
    keys_to_shard = [x.decode(encoding = 'utf-8') for x in all_redis_keys
                    if ((len(x.decode(encoding = 'utf-8').split('_')) == 3)
                    and (x.decode(encoding = 'utf-8').split('_')[2] <= end_date[4:6])
                    and (x.decode(encoding = 'utf-8').split('_')[2] >= start_date[4:6])
                    and (x.decode(encoding = 'utf-8').split('_')[0] == 'manulabels'))]
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

    return world_dict, shard_dict

def train(local_rank, world_size, world_dict, shard_dict, validation = False, Resume = None):
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        tsboard_path = "/home/wuzhihan/Projects/CNN/tensorboard_logs/reg_matrix_relu_v1"
        if not os.path.exists(tsboard_path):
            os.makedirs(tsboard_path)
        WRITER = SummaryWriter(log_dir=tsboard_path)

    model_path = '/home/wuzhihan/Projects/CNN/train_dir_0/model/CNN_param_reg_matrix_relu_v1'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if Resume:
        model_name = f'CNNLstmCLS_epoch_11_bs10000_sl64_ts3.pth.tar'
        model_data = torch.load(os.path.join(model_path, model_name), map_location='cpu')
        model_data['state_dict'] = ut.ddpModel_to_normal(model_data['state_dict'])

    LOGGER = logger.getLogger()
    LOGGER.setLevel('INFO') if local_rank == 0 else LOGGER.setLevel('WARNING')
    prior_epochs = 0 if not Resume else model_data['epoch'] + 1

    net = ConvLstmNet(in_features=INPUT_SIZE,
                      seq_len=SEQ_LEN,
                      out_features=OUTPUT_SIZE // 3).to(local_rank)


    if prior_epochs != 0:
        LOGGER.info(f"Prior epoch: {prior_epochs}, training resumes.")
        net.load_state_dict(model_data['state_dict'])
    else:
        LOGGER.info(f"No prior epoch, training start.")

    ddp_net = DDP(net, device_ids=[local_rank], output_device=local_rank)
    # optimizer = torch.optim.SGD(ddp_net.parameters(), lr = 1e-4)
    optimizer = torch.optim.AdamW(ddp_net.parameters(), lr = 1e-3)

    scheduler = lr_scheduler.StepLR(optimizer, step_size= 5, gamma=0.8)

    ddp_net.train()
    local_rank_id = world_dict[local_rank]
    local_rank_id.sort()
    len_train = int(len(local_rank_id) * 0.75)
    train_ids = local_rank_id[:len_train]
    val_ids = local_rank_id[len_train:]

    train_dataset = HFDatasetReg(local_ids = train_ids,
                                  shard_dict = shard_dict,
                                  batch_size=BATCH_SIZE,
                                  seq_len=SEQ_LEN,
                                  time_step = TIMESTEP)

    val_dataset = HFDatasetReg(local_ids=val_ids,
                                shard_dict=shard_dict,
                                batch_size=BATCH_SIZE,
                                seq_len=SEQ_LEN,
                                time_step=TIMESTEP)

    loss_fn = SelfMadeLoss(local_rank=local_rank)

    LOGGER.info(f'local rank {local_rank}: Dataset is prepared.')
    LOGGER.info(f'local rank {local_rank}: GPU calculating.')
    datalen = len(train_dataset)

    for epoch_idx in range(prior_epochs, EPOCHS):
        ddp_net.train()

        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_dataset):
            optimizer.zero_grad()
            h = net(x.permute(0,2,1).to(local_rank))
            y = y.to(local_rank)
            loss = loss_fn(h, y)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()


            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                rankic = spearman(y.detach().cpu(), h.detach().cpu())
                LOGGER.info(f"Epoch {epoch_idx}: {batch_idx + 1} / {datalen}), loss {loss.item()}, rank ic: {rankic}")

        if not (epoch_idx >= 20 and epoch_idx < 40):
            scheduler.step()

        if local_rank == 0:
            dist.barrier()
            WRITER.add_scalars(main_tag='LOSS--TRAIN',
                               tag_scalar_dict={
                                   'TOTAL': total_loss / datalen,
                               },
                               global_step=epoch_idx + 1)

            if validation:
                val_result = validate(net, val_dataset, loss_fn, local_rank)
                total_loss_val = val_result['total_loss']
                validation_rankic = val_result["rank_ic"]
                WRITER.add_scalars(main_tag='LOSS--EVAL',
                                   tag_scalar_dict={
                                       'TOTAL': total_loss_val,
                                   },
                                   global_step=epoch_idx + 1)

                LOGGER.info(f"Epoch {epoch_idx} validation loss: {total_loss_val}, rank ic: {validation_rankic}.")
            torch.save({
                'epoch': epoch_idx + 1,
                'train_loss': total_loss / datalen,
                'validation_result': val_result,
                'state_dict': ddp_net.state_dict(),
                'optimizer_dict': optimizer.state_dict(),
                'scheduler_dict': scheduler.state_dict()
            },
                os.path.join(model_path, f'CNNLstmCLS_epoch_{epoch_idx}_bs{BATCH_SIZE}_sl{SEQ_LEN}_ts{TIMESTEP}.pth.tar'))
            LOGGER.info(f'Epoch {epoch_idx} finished.')

        else:
            dist.barrier()

        del loss
        gc.collect()
        torch.cuda.empty_cache()

        LOGGER.info(f"Epoch {epoch_idx} is done.\n")

    LOGGER.info("Training has finished.")

    return




def validate(net, validation_dataset, lossfn, local_rank):

    net.eval()

    res = []

    total_loss = 0
    with torch.no_grad():
        for x, y in validation_dataset:

            h = net(x.permute(0, 2, 1).to(local_rank))
            res.append(spearman(y, h.detach().cpu()))
            loss = lossfn(h, y.to(local_rank))

            total_loss += loss.item()

    rankic = sum(res) / len(res)

    return {
        "total_loss": total_loss / len(validation_dataset),
        "rank_ic": rankic
    }



def main_train(LOGGER):

    train_start_date = '20210701'
    train_end_date = '20211031'
    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    LOGGER.info('Splitting indices for distributed training.')
    world_dict, shard_dict = distribute_to_child(start_date = train_start_date ,
                                                end_date = train_end_date,
                                                seq_len = SEQ_LEN,
                                                time_step = TIMESTEP,
                                                db = DB_ID,
                                                world_size = world_size)

    mp.spawn(train,
             args=(world_size, world_dict, shard_dict, True, None, ),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    main_train(Logger)