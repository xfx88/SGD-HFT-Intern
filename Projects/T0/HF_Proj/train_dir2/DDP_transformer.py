import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.multiprocessing as mp
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
import math
from tst import TransformerEncoder
from tst.utils import Optimiser
import tst.utilities as ut
from tst.utilities import factor_ret_cols
from src.dataset import HFDataset

import src.logger as logger

import os
import gc
import random
import pickle

from scipy.stats import spearmanr
from joblib import Parallel, delayed
from collections import OrderedDict

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9"

Logger = logger.getLogger()
Logger.setLevel("INFO")

# param1
# tOpt = ut.TrainingOptions(BATCH_SIZE=3,
#                           NUM_WORKERS=4,
#                           LR=1e-3,
#                           EPOCHS=80,
#                           N_stack=3,
#                           heads=4,
#                           query=8,
#                           value=8,
#                           d_model=128,
#                           d_input=43,
#                           d_output=4,
#                           attention_size = 50,
#                           window = 200,
#                           padding = 200,
#                           chunk_mode = 'window'
#                           )

# param2 按单个样本某个截面进行标准化（跨特征）， dim = 2
# tOpt = ut.TrainingOptions(BATCH_SIZE=3,
#                           NUM_WORKERS=4,
#                           LR=1e-3,
#                           EPOCHS=120,
#                           N_stack=7,
#                           heads=4,
#                           query=32,
#                           value=32,
#                           d_model=128,
#                           d_input=43,
#                           d_output=4,
#                           chunk_mode = None
#                           )

# param5 跨样本单特征标准化
tOpt = ut.TrainingOptions(BATCH_SIZE=3,
                          NUM_WORKERS=4,
                          LR=1e-3,
                          EPOCHS=80,
                          N_stack=7,
                          heads=4,
                          query=32,
                          value=32,
                          d_model=128,
                          d_input=42,
                          d_output=4,
                          chunk_mode = None
                          )

def get_samples(keys):
    maxlen = 4800
    res_list = []
    # Load dataset
    rs1 = ut.redis_connection(db=1)
    rs2 = ut.redis_connection(db=0)

    for key in keys:
        data = ut.read_data_from_redis(rs1, key)[factor_ret_cols].fillna(0)
        data = data.values.astype(np.float32)
        if len(data) == 0:
            continue
        # res_list.append(data)
        df_bytes = pickle.dumps(data)
        rs2.set(b'numpy' + b'_' + key, df_bytes)
    rs1.close()
    rs2.close()
    # return res_list

def get_all_data():
    rs = ut.redis_connection(db = 1)
    all_keys = rs.keys()
    rs.close()

    proc_num = 40
    batch_num = int(len(all_keys) / 40 + 1)
    keys_list = [all_keys[i * batch_num : (i + 1) * batch_num] for i in range(proc_num)]

    Parallel(n_jobs=proc_num, timeout=10000)(delayed(get_samples)(keys) for keys in keys_list)
    # data_array = deque()
    # for data in data_list:
    #     data_array.extend(data)
    pass
    # return data_array

def get_numpy_array(key):
    rs = ut.redis_connection(db = 0)
    data = ut.read_data_from_redis(rs, key)
    return data

def load_data(local_rank, world_size, epoch_idx, total_epoch, mode = 'train'):
    if not epoch_idx:
        epoch_idx = 0
    rs = ut.redis_connection(db = 0)
    world_dict = ut.read_data_from_redis(rs, f'DDP_{mode}_world_dict_{world_size}_epoch_{total_epoch}')
    keys = world_dict[epoch_idx][local_rank]

    data = []
    for k in keys:
        k_numpy = ut.read_data_from_redis(rs, k)
        if len(k_numpy) > 4800:
            continue
        data.append(k_numpy)
    rs.close()

    return data

def train_val_splitter(dataset, epoch_idx, percent = 0.85, validation = True):
    len_train = int(len(dataset) * percent)
    len_valid = len(dataset) - len_train
    train_set, val_set = random_split(dataset, [len_train, len_valid], generator=torch.Generator().manual_seed(epoch_idx))
    if not validation:
        del val_set
        return train_set, None
    return train_set, val_set

def ddpModel_to_normal(ddp_state_dict: OrderedDict):
    new_state_dict = OrderedDict()
    for k, v in ddp_state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict

class TickMSE(nn.Module):
    def __init__(self, scale_factor, local_rank, reduction = 'mean'):
        super().__init__()
        self.scale_factor = scale_factor
        self.base_loss = nn.SmoothL1Loss(reduction=reduction)

    def forward(self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor) -> torch.Tensor:
        # y_true = y_true.reshape(20, 25, 200, -1).reshape(500, 200, -1)[:, -1, :]
        # y_pred = y_pred.reshape(20, 25, 200, -1).reshape(500, 200, -1)[:, -1, :]
        loss = self.scale_factor * (self.base_loss(y_pred, y_true))

        return loss


def train(local_rank, world_size, validation = False, Resume = None):
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    model_path = 'model/transformer/param_5/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if Resume:
        model_name = 'ddpTransformer_epoch_5.pth.tar'
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
    loss_function = TickMSE(scale_factor=1e3, local_rank = local_rank).to(local_rank)
    # loss_function = nn.MSELoss(reduction = 'mean')
    opt = Optimiser(ddp_net, scale_factor = 1e-3, warmup_steps=8000)
    # opt = Optimiser(ddp_net, scale_factor = 1e-3, warmup_steps=4000)
    # opt = optim.AdamW(ddp_net.parameters())
    if prior_epochs != 0:
        opt.optimiser.load_state_dict(model_data['optimizer_dict'])
        opt.current_step = model_data['current_step']

    ddp_net.train()

    data = load_data(local_rank, world_size, prior_epochs, tOpt.EPOCHS)
    dataset = HFDataset(data, tick_num=tOpt.window, LEN_SEQ=100)

    del data
    gc.collect()

    ACCUMULATED_STEP = 2

    LOGGER.info(f'local rank {local_rank}: Dataset is prepared.')
    LOGGER.info(f'local rank {local_rank}: GPU calculating.')
    train_dataset, validation_dataset = train_val_splitter(dataset, 0, validation=validation)
    datalen = len(train_dataset)
    for epoch_idx in range(prior_epochs, tOpt.EPOCHS):
        total_loss = 0.0
        if epoch_idx == 15:
            opt.scale_factor = 0.005
        for batch_idx, (x, y) in enumerate(train_dataset):
            opt.optimiser.zero_grad()
            # for i in range(5):
            #     opt.optimiser.zero_grad()
            #     netout = net(x[i * 100 : (i + 1) * 100, ...].to(local_rank))
            #     # Comupte loss
            #     loss = loss_function(netout, y[i * 100 : (i + 1) * 100, ...].to(local_rank))
            #     # Backpropage loss
            #     loss.backward()
            #     total_loss += loss.item() / 5
            #     opt.step()

            netout = net(x.to(local_rank))
            # Comupte loss
            loss = loss_function(netout, y.to(local_rank))
            total_loss += loss.item()
            # Backpropage loss
            loss.backward()
            opt.step()




            # if (batch_idx + 1) % ACCUMULATED_STEP == 0:
            # Update weights



            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                LOGGER.info(f'Epoch {epoch_idx}: {batch_idx + 1} / {datalen}), loss {loss.item()}.')

        if local_rank == 0:
            dist.barrier()
            if validation:
                val_loss = validate(net, validation_dataset, loss_function, local_rank)
                LOGGER.info(f'Epoch {epoch_idx} validation loss: {val_loss}.')
            torch.save({
                'epoch': epoch_idx,
                'train_loss': total_loss / datalen,
                'validation_loss': val_loss,
                'state_dict': ddp_net.state_dict(),
                'current_step': opt.current_step,
                'optimizer_dict': opt.optimiser.state_dict(),
            }, os.path.join(model_path, f'ddpTransformer_epoch_{epoch_idx}.pth.tar'))
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

    running_loss = 0
    with torch.no_grad():
        for x, y in validation_dataset:
            # netout = net(x[i * 10: (i + 1) + 10, ...].to(local_rank))
            netout = net(x.to(local_rank))
            # Comupte loss
            loss = loss_function(netout, y.to(local_rank))
            running_loss += loss.item()
    return running_loss / len(validation_dataset)


def generate_process_keys(world_size, epoch, start_date, end_date, mode = 'train'):
    world_dict = {}
    rs = ut.redis_connection(db = 0)
    all_redis_keys = rs.keys()
    if bytes(f'DDP_{mode}_world_dict_{world_size}_epoch_{epoch}', encoding = 'utf-8') in all_redis_keys:
        return

    keys_to_dist = [x for x in all_redis_keys
                    if ((len(str(x).split('_')) == 2)
                    and (int(str(x).split('_')[1][:8]) <= end_date)
                    and (int(str(x).split('_')[1][:8]) >= start_date)
                    and (str(x).split('_')[0][2:] == 'numpy'))]
    if mode == 'train':
        len_part = math.ceil(len(keys_to_dist) / world_size)
        keys_indices = list(range(len(keys_to_dist)))
        for epoch_idx in range(epoch):
            random.seed(epoch_idx)
            random.shuffle(keys_indices)
            world_dict[epoch_idx] = {i: [keys_to_dist[j] for j in keys_indices[i * len_part: (i + 1) * len_part]]
                                     for i in range(world_size)}
    elif mode == 'test':
        len_part = int(len(keys_to_dist) / world_size + 1)
        keys_indices = list(range(len(keys_to_dist)))
        keys_to_dist.sort()
        world_dict[0] = {i: [keys_to_dist[j] for j in keys_indices[i * len_part: (i + 1) * len_part]]
                         for i in range(world_size)}

    world_dict_bytes = pickle.dumps(world_dict)
    rs.set(f'DDP_{mode}_world_dict_{world_size}_epoch_{epoch}', world_dict_bytes)
    rs.close()
    return



def main_train(LOGGER):

    # prior_epochs = len(os.listdir(model_path))
    # model_name = f'ddpTransformer_epoch_{prior_epochs - 1}.pth.tar'
    # model_data = torch.load(model_path + model_name)

    train_start_date = 20210701
    train_end_date = 20210930
    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    LOGGER.info('Splitting indices for distributed training.')
    generate_process_keys(world_size, tOpt.EPOCHS, train_start_date, train_end_date)

    # world_size = 1
    mp.spawn(train,
             args=(world_size, True, None, ),
             nprocs=world_size,
             join=True)



if __name__ == '__main__':
    # main_test(Logger)
    # warmup_train(Logger)
    main_train(Logger)
    # main_resume_train(Logger)