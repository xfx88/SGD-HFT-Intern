import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.multiprocessing as mp
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm

from tst import Transformer
from tst.utils import Optimiser
import tst.utilities as ut
from tst.utilities import factor_ret_cols
from src.dataset import HFDataset

import src.logger as logger
import logging

import os
import gc
import random
import pickle

from scipy.stats import spearmanr
from joblib import Parallel, delayed
from collections import OrderedDict

from contextlib import nullcontext

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,5,6,9"

Logger = logger.getLogger()
Logger.setLevel("INFO")

# param2
# tOpt = ut.TrainingOptions(BATCH_SIZE=10,
#                           NUM_WORKERS=4,
#                           LR=5*1e-5,
#                           EPOCHS=80,
#                           N_stack=2,
#                           heads=4,
#                           query=16,
#                           value=16,
#                           d_model=64,
#                           d_input=43,
#                           d_output=1,
#                           attention_size = 80,
#                           window = 160,
#                           padding = 160//4,
#                           chunk_mode = 'window'
#                           )
# #param3
# tOpt = ut.TrainingOptions(BATCH_SIZE=10,
#                           NUM_WORKERS=4,
#                           LR=1*1e-5,
#                           EPOCHS=200,
#                           N_stack=2,
#                           heads=4,
#                           query=8,
#                           value=8,
#                           d_model=128,
#                           d_input=43,
#                           d_output=1,
#                           attention_size = 50,
#                           window = 200,
#                           padding = 200//4,
#                           chunk_mode = 'window'
#                           )

# param4
tOpt = ut.TrainingOptions(BATCH_SIZE=8,
                          NUM_WORKERS=4,
                          LR=1e-3,
                          EPOCHS=100,
                          N_stack=3,
                          heads=2,
                          query=8,
                          value=8,
                          d_model=128,
                          d_input=43,
                          d_output=1,
                          attention_size = 50,
                          window = 200,
                          padding = 50,
                          chunk_mode = 'window'
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
    world_dict = ut.read_data_from_redis(rs, f'{mode}_world_dict_{world_size}_epoch_{total_epoch}')
    keys = world_dict[epoch_idx][local_rank]

    data = []
    for k in keys:
        k_numpy = ut.read_data_from_redis(rs, k)
        if len(k_numpy) > 4800:
            continue
        data.append(k_numpy)
    rs.close()

    return data

def train_val_splitter(data, percent = 0.85, validation = True):
    dataset = HFDataset(data, tick_num=tOpt.window)
    if not validation:
        return dataset, None

    len_train = int(len(dataset) * percent)
    len_valid = len(dataset) - len_train
    train_set, val_set = random_split(dataset, [len_train, len_valid])

    return train_set, val_set

def ddpModel_to_normal(ddp_state_dict: OrderedDict):
    new_state_dict = OrderedDict()
    for k, v in ddp_state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict

class TickMSE(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.base_loss = nn.MSELoss()

    def forward(self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor) -> torch.Tensor:

        return self.scale_factor * self.base_loss(y_pred, y_true)


def train(local_rank, world_size, validation = False, Resume = None):
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    model_path = 'model/transformer/param_comb_4/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if Resume:
        model_name = 'ddpTransformer_epoch_79.pth.tar'
        model_data = torch.load(os.path.join(model_path, model_name), map_location='cpu')
        model_data['state_dict'] = ddpModel_to_normal(model_data['state_dict'])

    LOGGER = logger.getLogger()
    LOGGER.setLevel('INFO') if local_rank == 0 else LOGGER.setLevel('WARNING')
    prior_epochs = 0 if not Resume else model_data['epoch'] + 1

    net = Transformer(tOpt).to(local_rank)


    if prior_epochs != 0:
        LOGGER.info(f"Prior epoch: {prior_epochs}, training resumes.")
        net.load_state_dict(model_data['state_dict'])
        optimizer.load_state_dict(model_data['optimizer_dict'])
    else:
        LOGGER.info(f"No prior epoch, training start.")

    ddp_net = DDP(net, device_ids=[local_rank], output_device=local_rank)
    loss_function = TickMSE(scale_factor=1e6).to(local_rank)
    opt = Optimiser(ddp_net, scale_factor=5e-2, warmup_steps=4000)

    ddp_net.train()

    LOGGER.info(f'local rank {local_rank}: GPU calculating.')
    for epoch_idx in range(prior_epochs, tOpt.EPOCHS):
        data = load_data(local_rank, world_size, epoch_idx, tOpt.EPOCHS)
        train_dataset, validation_dataset = train_val_splitter(data, validation=validation)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=tOpt.BATCH_SIZE,
                                      num_workers=tOpt.NUM_WORKERS)
        if validation:
            validation_dataloader = DataLoader(validation_dataset,
                                               batch_size=tOpt.BATCH_SIZE,
                                               num_workers=tOpt.NUM_WORKERS,
                                               shuffle=False)
        # train_sampler.set_epoch(epoch_idx)
        for batch_idx, (x, y) in enumerate(train_dataloader):
            opt.optimiser.zero_grad()
            # Propagate input
            netout = net(x.to(local_rank))
            # Comupte loss
            loss = loss_function(netout, y.to(local_rank))
            # Backpropage loss
            loss.backward()
            # Update weights
            opt.step()
            if batch_idx % 20 == 0:
                LOGGER.info(f'Epoch {epoch_idx}, batch {batch_idx}: {loss.item()}.')
        del loss
        gc.collect()
        torch.cuda.empty_cache()

        if local_rank == 0:
            dist.barrier()
            torch.save({
                'epoch': epoch_idx,
                'state_dict': ddp_net.state_dict(),
                'optimizer_dict': opt.optimiser.state_dict(),
            }, os.path.join(model_path, f'ddpTransformer_epoch_{epoch_idx}.pth.tar'))
            LOGGER.info(f'Epoch {epoch_idx} finished.')
            if validation:
                val_loss = validate(net, validation_dataloader, loss_function, local_rank)
                LOGGER.info(f'Epoch {epoch_idx} validation loss: {val_loss}.')
        else:
            dist.barrier()
        del data, train_dataset, train_dataloader
        gc.collect()
        LOGGER.info(f"Epoch {epoch_idx} is done.\n")

    LOGGER.info("Training has finished.")

    return

def validate(net, validation_dataloader, loss_function, local_rank):

    running_loss = 0
    with torch.no_grad():
        for x, y in validation_dataloader:
            netout = net(x.to(local_rank)).cpu()
            running_loss += loss_function(y, netout)
    return running_loss / len(validation_dataloader)

def evaluate(local_rank, world_size):
    LOGGER = logger.getLogger()
    LOGGER.setLevel('INFO') if local_rank == 0 else LOGGER.setLevel('WARNING')
    model_path = 'model/transformer/param_comb_2/'
    # model_name_list = os.listdir(model_path)
    # model_name_list.sort()
    model_name_list = ['ddpTransformer_epoch_69.pth.tar']
    model_data = torch.load(model_path + model_name_list[0])

    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    net = Transformer(tOpt).to(local_rank)
    net = DDP(net, device_ids = [local_rank], output_device = local_rank)
    net.load_state_dict(model_data['state_dict'])

    net.eval()

    y_pred_all = []
    y_all = []

    LOGGER.info("Fetching test data ...")
    data = load_data(local_rank, world_size, epoch_idx = None, total_epoch = 1, mode = 'test')
    test_dataset = HFDataset(data, tick_num = tOpt.window)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=30,
                                 num_workers=tOpt.NUM_WORKERS,
                                 shuffle = False)
    LOGGER.info("Test Dataloader has been prepared.")

    with torch.no_grad():
        for x, y in tqdm(test_dataloader, total=len(test_dataloader)):
            y_pred = net(x.to(local_rank)).to('cpu')
            y_pred_all.append(y_pred.detach().numpy())
            y_all.append(y.detach().numpy())
            torch.cuda.empty_cache()
    LOGGER.info("Prediction has finished.")
    y_pred_all = np.concatenate(y_pred_all)
    y_all = np.concatenate(y_all)
    y_pred_all = y_pred_all.reshape(-1,1)
    y_all = y_all.reshape(-1,1)
    spm = spearmanr(y_pred_all, y_all)
    LOGGER.info(f'spearman correlation: {spm}')
    # metrics = {
    #     'training_loss': lambda y_true, y_pred: OZELoss(alpha=0.3, reduction='none')(y_true, y_pred).numpy(),
    #     'mse_tint_total': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[-1], reduction='none'),
    #     'mse_cold_total': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[0, 1, 2, 3, 4, 5, 6], reduction='none'),
    #     'mse_tint_occupation': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[-1], reduction='none', occupation=occupation),
    #     'mse_cold_occupation': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[0, 1, 2, 3, 4, 5, 6], reduction='none', occupation=occupation),
    #     'r2_tint': lambda y_true, y_pred: np.array([r2_score(y_true[:, i, -1], y_pred[:, i, -1]) for i in range(y_true.shape[1])]),
    #     'r2_cold': lambda y_true, y_pred: np.array([r2_score(y_true[:, i, 0:-1], y_pred[:, i, 0:-1]) for i in range(y_true.shape[1])])
    # }
    #
    # logger = Logger(f'data/logs/training.csv', model_name=net.name,
    #                 params=[y for key in metrics.keys() for y in (key, key+'_std')])

    # Switch to evaluation
    # _ = net.eval()

    # results_metrics = {
    #     key: value for key, func in metrics.items() for key, value in {
    #         key: func(y_true, predictions).mean(),
    #         key+'_std': func(y_true, predictions).std()
    #     }.items()
    # }

def generate_process_keys(world_size, epoch, start_date, end_date, mode = 'train'):
    world_dict = {}
    rs = ut.redis_connection(db = 0)
    all_redis_keys = rs.keys()
    if bytes(f'{mode}_world_dict_{world_size}_epoch_{epoch}', encoding = 'utf-8') in all_redis_keys:
        return

    keys_to_dist = [x for x in all_redis_keys if len()
                    and (str(x).split('_')[1].isdigit())
                    and (int(str(x).split('_')[1]) <= end_date)
                    and (int(str(x).split('_')[1]) >= start_date)]
    if mode == 'train':
        len_part = int(len(keys_to_dist) / world_size + 1)
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
    rs.set(f'{mode}_world_dict_{world_size}_epoch_{epoch}', world_dict_bytes)
    rs.close()
    return

def main_resume_train(LOGGER):
    """
    params for method "train":
    local_rank, world_size, validation = False, model_data = None
    """
    train_start_date = 20210701
    train_end_date = 20210731
    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    LOGGER.info('Splitting indices for distributed training.')
    generate_process_keys(world_size, tOpt.EPOCHS, train_start_date, train_end_date)


    # world_size = 1
    mp.spawn(train,
             args=(world_size, False, True, ),
             nprocs=world_size,
             join=True)

def main_train(LOGGER):

    # prior_epochs = len(os.listdir(model_path))
    # model_name = f'ddpTransformer_epoch_{prior_epochs - 1}.pth.tar'
    # model_data = torch.load(model_path + model_name)

    train_start_date = 20210701
    train_end_date = 20210931
    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    LOGGER.info('Splitting indices for distributed training.')
    generate_process_keys(world_size, tOpt.EPOCHS, train_start_date, train_end_date)

    # world_size = 1
    mp.spawn(train,
             args=(world_size, False, None, ),
             nprocs=world_size,
             join=True)

def main_test(LOGGER):
    test_start_date = 20211001
    test_end_date = 20211031
    world_size = 1
    generate_process_keys(world_size, 1, test_start_date, test_end_date, mode = 'test')

    mp.spawn(evaluate,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    # main_test(Logger)
    main_train(Logger)
    # main_resume_train(Logger)