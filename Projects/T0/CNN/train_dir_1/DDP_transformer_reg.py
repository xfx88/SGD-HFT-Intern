import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import math
from tst import TransformerEncoder
import tst.utilities as ut
from tst.utilities import factor_ret_cols
from src.dataset_reg import HFDataset
import torchsort

import src.logger as logger

import os
import gc
import random
import pickle

from scipy.stats import spearmanr
from joblib import Parallel, delayed
from collections import OrderedDict

GLOBAL_SEED = 12309
DB_ID = 1
WARMUP = 500
BATCH_SIZE = 1
SEQ_LEN = 64
INPUT_SIZE = 44
OUTPUT_SIZE = 3
EPOCHS = 50

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12359"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["OMP_NUM_THREADS"] = "8"
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

Logger = logger.getLogger()
Logger.setLevel("INFO")

# class Optimiser:
#
#     def __init__(self, model, optimiser=None, scale_factor=1.0, warmup_steps=2000, beta1=0.9, beta2=0.98, epsilon=1e-9):
#
#         if optimiser is not None: self.optimiser = optimiser
#         # else: self.optimiser = torch.optim.Adam(model.parameters(), lr=0, betas=(beta1, beta2), eps=epsilon)
#         else: self.optimiser = torch.optim.SGD(model.parameters(), lr = 0)
#         self.scale_factor = scale_factor
#         self.warmup_steps = math.pow(warmup_steps, -1.5)
#         self.current_step = 0
#         self.inv_sqrt_d_input = math.pow(model.module.d_input, -0.5)
#
#         self.lrate = lambda step: self.inv_sqrt_d_input * min(math.pow(step, -0.5), step * self.warmup_steps)
#         self.rate = None
#
#     def step(self):
#         self.current_step += 1
#         self.rate = self.scale_factor * self.lrate(self.current_step)
#         for i in self.optimiser.param_groups: i['lr'] = self.rate
#         self.optimiser.step()
#
#     def zero_grad(self):
#         self.optimiser.zero_grad()

def corrcoef(target, pred):
    pred_n = pred - pred.mean() + torch.rand(pred.shape).cuda() * 1e-9
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return (pred_n * target_n).sum()

def spearman(
    pred,
    target,
    regularization="l2",
    regularization_strength=1.0,
    weight = (1/3, 1/3, 1/3)):

    weighted_corr = 0
    corr_list = []
    for i in range(pred.shape[-1]):
        pred_i = pred[:, :, i : i + 1].flatten().unsqueeze(-1).T
        target_i = target[:, :, i : i + 1].flatten().unsqueeze(-1).T
        pred_i = torchsort.soft_rank(
                 pred_i,
                 regularization=regularization,
                 regularization_strength=regularization_strength,)
        corr = corrcoef(target_i, pred_i)
        weighted_corr += weight[i] * (1 - corr)
        corr_list.append(round(corr.item(), 4))
    return weighted_corr, torch.tensor(corr_list)

class JointLoss(nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()
        self.L1loss = nn.SmoothL1Loss()
        self._mode = "train"
        self._ic_list = []

    def forward(self, pred: torch.Tensor, target: torch.Tensor):

        self.spearman_loss, self.ic = spearman(pred = pred, target = target)
        if self._mode == "eval":
            self._ic_list.append(self.ic)
        l1 = self.L1loss(pred, target)

        return 1e3 * (self.spearman_loss * (l1 + 1))

    def train(self):
        self._ic_list.clear()
        self._mode ="train"

    def eval(self):
        self._mode = "eval"

    @property
    def rank_ic(self):
        return self.ic.data

    @property
    def eval_rank_ic(self):
        return torch.cat(self._ic_list).mean(0)

def shard_keys(start_date, end_date, seq_len = 50, time_step = 1, db = DB_ID):
    shard_dict_whole = dict()
    rs = ut.redis_connection(db=db)
    all_redis_keys = rs.keys()
    keys_to_shard = [x.decode(encoding = 'utf-8') for x in all_redis_keys
                    if ((len(x.decode(encoding = 'utf-8').split('_')) == 3)
                    and (x.decode(encoding = 'utf-8').split('_')[2] <= end_date[4:6])
                    and (x.decode(encoding = 'utf-8').split('_')[2] >= start_date[4:6])
                    and (x.decode(encoding = 'utf-8').split('_')[0] == 'reg'))]
    cnt = 0 # 记录所有序列的长度
    keys_to_shard.sort()
    for key in keys_to_shard:
        shard_dict_whole[cnt] = key
        cnt += 1
    rs.close()

    return shard_dict_whole, cnt


def distribute_to_child(start_date, end_date, seq_len = 50, db = DB_ID, world_size = 1):
    shard_dict, key_num = shard_keys(start_date, end_date, seq_len = seq_len, time_step = seq_len, db = db)
    idx_list = list(range(key_num))
    len_part = math.ceil(key_num / world_size)
    random.Random(GLOBAL_SEED).shuffle(idx_list)
    world_dict = {i: [idx_list[j] for j in idx_list[i * len_part: (i + 1) * len_part]]
                  for i in range(world_size)}

    return world_dict, shard_dict

# param5 跨样本单特征标准化
tOpt = ut.TrainingOptions(BATCH_SIZE=BATCH_SIZE,
                          EPOCHS=EPOCHS,
                          N_stack=2,
                          heads=4,
                          query=8,
                          value=8,
                          d_model=192,
                          d_input=INPUT_SIZE,
                          d_output=OUTPUT_SIZE,
                          chunk_mode = None,
                          pe = "regular"
                          )

def ddpModel_to_normal(ddp_state_dict: OrderedDict):
    new_state_dict = OrderedDict()
    for k, v in ddp_state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict


def train(local_rank, world_size, world_dict, shard_dict, validation = False, Resume = None):
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        tsboard_path = "/home/wuzhihan/Projects/CNN/tensorboard_logs/transformer_reg"
        if not os.path.exists(tsboard_path):
            os.makedirs(tsboard_path)
        WRITER = SummaryWriter(log_dir=tsboard_path)

    model_path = '/home/wuzhihan/Projects/CNN/train_dir_1/transformer/param_reg/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if Resume:
        model_name = f'TST_epoch_3_bs5000_sl64.pth.tar'
        model_data = torch.load(os.path.join(model_path, model_name), map_location='cpu')
        model_data['state_dict'] = ddpModel_to_normal(model_data['state_dict'])

    LOGGER = logger.getLogger()
    LOGGER.setLevel('INFO') if local_rank == 0 else LOGGER.setLevel('WARNING')
    prior_epochs = 0 if not Resume else model_data['epoch'] + 1

    net = TransformerEncoder(tOpt).to(local_rank)

    if prior_epochs != 0:
        LOGGER.info(f"Prior epoch: {prior_epochs}, training resumes.")
        net.load_state_dict(model_data['state_dict'])
    else:
        LOGGER.info(f"No prior epoch, training start.")

    ddp_net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    loss_fn = JointLoss().to(local_rank)
    # loss_mse = nn.MSELoss()

    optimizer = torch.optim.Adam(ddp_net.parameters(), lr = 1e-5)
    if prior_epochs != 0:
        optimizer.load_state_dict(model_data['optimizer_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    ddp_net.train()
    loss_fn.train()

    local_rank_id = world_dict[local_rank]
    local_rank_id.sort()
    len_train = int(len(local_rank_id) * 0.7)
    train_ids = local_rank_id[:len_train]
    val_ids = local_rank_id[len_train:]

    train_dataset = HFDataset(local_ids=train_ids,
                              shard_dict=shard_dict,
                              batch_size=BATCH_SIZE,
                              seq_len=SEQ_LEN)

    val_dataset = HFDataset(local_ids=val_ids,
                               shard_dict=shard_dict,
                               batch_size=BATCH_SIZE,
                               seq_len=SEQ_LEN)


    LOGGER.info(f'local rank {local_rank}: Dataset is prepared.')
    LOGGER.info(f'local rank {local_rank}: GPU calculating.')
    datalen = len(train_dataset)
    for epoch_idx in range(prior_epochs, EPOCHS):
        ddp_net.train()
        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_dataset):
            optimizer.zero_grad()
            y_pred = net(x.to(local_rank))

            loss = loss_fn(y_pred, y.to(local_rank))
            # loss2 = loss_mse(y_pred, y.to(local_rank))

            total_loss += loss.item()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1e3)
            optimizer.step()

            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                LOGGER.info(f"Epoch {epoch_idx}: {batch_idx + 1} / {datalen}), loss {round(loss.item(),3)}, IC: {loss_fn.rank_ic}")

        del x, y
        gc.collect()
        scheduler.step()

        if local_rank == 0:
            dist.barrier()

            for name, param in ddp_net.named_parameters():
                WRITER.add_histogram(name, param.clone().cpu().data.numpy(), epoch_idx + 1)
                WRITER.add_histogram(name + '/grad', param.grad.clone().cpu().data.numpy(), epoch_idx + 1)

            if validation:
                val_loss = validate(net, val_dataset, loss_fn, local_rank)
                LOGGER.info(f'Epoch {epoch_idx} validation loss: {val_loss}, IC: {loss_fn.eval_rank_ic}.')
            torch.save({
                'epoch': epoch_idx,
                'train_loss': total_loss / datalen,
                'validation_loss': val_loss,
                'state_dict': ddp_net.state_dict(),
                'optimizer_dict': optimizer.state_dict(),
                'scheduler_dict': scheduler.state_dict
            }, os.path.join(model_path, f'TST_epoch_{epoch_idx}_bs{BATCH_SIZE}_sl{SEQ_LEN}.pth.tar'))
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
    net.eval()
    loss_function.eval()
    running_loss = 0
    with torch.no_grad():
        for x, y in validation_dataset:
            netout = net(x.to(local_rank))

            loss = loss_function(netout, y.to(local_rank))
            running_loss += loss.item()
    return running_loss / len(validation_dataset)


def main_train(LOGGER):

    train_start_date = '20210701'
    train_end_date = '20211031'
    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    LOGGER.info('Splitting indices for distributed training.')
    world_dict, shard_dict = distribute_to_child(start_date = train_start_date ,
                                                end_date = train_end_date,
                                                seq_len = SEQ_LEN,
                                                db = DB_ID,
                                                world_size = world_size)

    # world_size = 1
    mp.spawn(train,
             args=(world_size, world_dict, shard_dict, True, None, ),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    main_train(Logger)