
import sys
sys.path.append("/home/yby/SGD-HFT-Intern/Projects/T0/CNN")
import os
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
os.environ["OMP_NUM_THREADS"] = '4'
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,5,6,7,8,9"
import torch.multiprocessing as mp

from torch import distributed as dist
from torchsampler import ImbalancedDatasetSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import utilities as ut

import math
import random

import gc
import src.logger as logger
from src.dataset_cls import HFDataset, HFDatasetVal
from utilities import *



DB_ID = 0

Logger = logger.getLogger()
Logger.setLevel("INFO")


GLOBAL_SEED = 2098
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

BATCH_SIZE = 5000
WARMUP = 1000
RESUME = None

INPUT_SIZE = 44
OUTPUT_SIZE = 9
SEQ_LEN = 64
TIMESTEP = 5

EPOCHS = 80

validation_dates = ['20211020','20211021','20211022','20211025','20211026','20211027','20211028','20211029']

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int,
                       out_channels: int,
                       kernel_size: tuple,
                       padding: tuple,
                       dilation: tuple = (1, )):

        super(CNNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              dilation=dilation,
                              kernel_size=kernel_size,
                              padding = padding)
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv(x)
        x = self.relu(self.bn(x))

        return x

class ConvLstmNet(nn.Module):
    def __init__(self, in_features, seq_len, out_features):
        super(ConvLstmNet, self).__init__()
        # set size
        self.in_features = in_features
        self.seq_len     = seq_len
        self.out_features = out_features

        self.conv_0 =  CNNBlock(in_channels= in_features,out_channels = 64,kernel_size=(3, ),dilation=(4,),padding=(4, ))
        self.conv_1 =  CNNBlock(in_channels= 64,out_channels = 128,kernel_size=(5, ),padding=(2, ))
        self.conv_2 = CNNBlock(in_channels=128,out_channels=256,kernel_size=(7,),padding=(3,))
        self.conv_3 = CNNBlock(in_channels=256,out_channels=512,kernel_size=(5,),padding=(2,))

        self.downsample = nn.Conv1d(in_channels=512,out_channels=in_features,kernel_size=(3,),padding = (1,)) # N, 44, 64

        self.out_fcn0 = nn.Linear(in_features=seq_len, out_features=1)
        self.out_fcn1 = nn.Linear(in_features=in_features, out_features=out_features)

        self.relu = nn.ReLU()

        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight.data, nonlinearity='conv1d')
                nn.init.zeros_(layer.bias.data)
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight.data, nonlinearity='linear')
                nn.init.zeros_(layer.bias.data)

    def residual_connect(self, x, res):
        return self.relu(self.downsample(x) + res)

    def forward(self, x):
        res = x
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.residual_connect(x, res)

        x = self.out_fcn0(x).squeeze(-1)
        y = self.out_fcn1(x)

        return y


class Optimiser:

    def __init__(self, model, optimiser=None, scale_factor=1.0, warmup_steps=2000, beta1=0.9, beta2=0.98, epsilon=1e-9):

        if optimiser is not None: self.optimiser = optimiser
        else: self.optimiser = torch.optim.Adam(model.parameters(), lr=0, betas=(beta1, beta2), eps=epsilon)
        # else: self.optimiser = torch.optim.SGD(model.parameters(), lr = 0)
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

    def zero_grad(self):
        self.optimiser.zero_grad()


def shard_keys(start_date, end_date, seq_len = 50, time_step = 1, db = DB_ID):
    shard_dict_whole = dict()
    rs = ut.redis_connection(db=db)
    all_redis_keys = rs.keys()
    keys_to_shard = [x.decode(encoding = 'utf-8') for x in all_redis_keys
                    if ((len(x.decode(encoding = 'utf-8').split('_')) == 3)
                    and (x.decode(encoding = 'utf-8').split('_')[2] <= end_date[4:6])
                    and (x.decode(encoding = 'utf-8').split('_')[2] >= start_date[4:6])
                    and (x.decode(encoding = 'utf-8').split('_')[0] == 'clslabels'))]
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
        tsboard_path = "/home/yby/SGD-HFT-Intern/Projects/T0/CNN/tensorboard_logs/cls5_v2"
        if not os.path.exists(tsboard_path):
            os.makedirs(tsboard_path)
        WRITER = SummaryWriter(log_dir=tsboard_path)
    model_path = '/home/yby/SGD-HFT-Intern/Projects/T0/CNN/train_dir_0/model/CNN/param_cls18_v2_adam_datamodified/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if Resume:
        model_name = f'CNNLstmCLS_epoch_1_bs5000_sl64_ts5.pth.tar'
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
    # loss_function = nn.CrossEntropyLoss(weight=loss_weight).to(local_rank)
    # lossfn_p2 = nn.CrossEntropyLoss(weight=torch.tensor([1.,25.,25.])).cuda()
    lossfn_p5 = nn.CrossEntropyLoss(weight = torch.tensor([20/22, 1/22, 1/22])).cuda()
    # lossfn_p18 = nn.CrossEntropyLoss(weight=torch.tensor([2.,7.,7.])).cuda()
    # lossfn_p2 = FocalLoss(class_num = OUTPUT_SIZE // 3, gamma = 4)
    # lossfn_p5 = FocalLoss(class_num = OUTPUT_SIZE // 3, gamma = 2)
    # lossfn_p18 = FocalLoss(class_num = OUTPUT_SIZE // 3, gamma = 4)
    # optimiser = torch.optim.Adam(ddp_net.parameters(), lr = 1e-4)
    opt = Optimiser(ddp_net, scale_factor = 1e-3, warmup_steps=WARMUP)
    # opt = torch.optim.SGD(ddp_net.parameters(), lr = 1e-5)
    # opt = Optimiser(model = ddp_net, optimiser=opt)
    if prior_epochs != 0:
        opt.optimiser.load_state_dict(model_data['optimizer_dict'])
        # opt.current_step = model_data['current_step']

    ddp_net.train()
    local_rank_id = world_dict[local_rank]
    local_rank_id.sort()
    len_train = int(len(local_rank_id) * 0.75)
    train_ids = local_rank_id[:len_train]
    val_ids = local_rank_id[len_train:]
    train_dataset = HFDataset(local_ids = train_ids,
                              shard_dict = shard_dict,
                              seq_len=SEQ_LEN,
                              time_step = TIMESTEP)
    targets = train_dataset.get_labels().long()
    class_count = torch.unique(targets, return_counts = True)[1]
    print(class_count)
    weight = 1. / class_count
    sample_weight = weight[targets]

    sampler = WeightedRandomSampler(sample_weight, len(train_dataset), replacement=False)
    train_dataloader = DataLoader(dataset = train_dataset,
                                  batch_size = BATCH_SIZE,
                                  sampler = sampler,
                                  num_workers=0)
    # train_dataloader = DataLoader(dataset = train_dataset,
    #                               batch_size = BATCH_SIZE,
    #                               sampler = ImbalancedDatasetSampler(train_dataset),
    #                               num_workers=0,
    #                               )

    val_dataset = HFDatasetVal(local_ids=val_ids,
                               shard_dict=shard_dict,
                               batch_size=BATCH_SIZE,
                               seq_len=SEQ_LEN,
                               time_step=TIMESTEP)

    del train_dataset
    gc.collect()

    LOGGER.info(f'local rank {local_rank}: Dataset is prepared.')
    LOGGER.info(f'local rank {local_rank}: GPU calculating.')
    datalen = len(train_dataloader)
    for epoch_idx in range(prior_epochs, EPOCHS):
        ddp_net.train()

        # total_loss_p2 = 0.0
        total_loss_p5 = 0.0
        # total_loss_p18 = 0.0
        total_loss = 0.0
        DENOMINATOR = 2

        if epoch_idx != 0 and epoch_idx % 3 == 0:
            opt.scale_factor /= DENOMINATOR
            DENOMINATOR -= 0.02

        for batch_idx, (x, y) in enumerate(train_dataloader):
            opt.zero_grad()
            h_p5 = net(x.permute(0,2,1).to(local_rank))

            # loss_p2 = lossfn_p2(h_p2, y[..., 0].cuda().long())
            loss_p5 = lossfn_p5(h_p5, y.cuda().long())
            # loss_p18 = lossfn_p18(h_p18, y[..., 2].cuda().long())
            # total_loss_p2 += loss_p2.item()
            total_loss_p5 += loss_p5.item()
            # total_loss_p18 += loss_p18.item()

            loss = loss_p5
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm = 1e7)
            opt.step()

            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                class_count = torch.unique(y, return_counts=True)[1]
                LOGGER.info(f"Epoch {epoch_idx}: {batch_idx + 1} / {datalen}), loss {loss.item()}. ")
                            # f" [p2: {loss_p2.item()}, p5: {loss_p5.item()}, p18: {loss_p18.item()}].")

        if local_rank == 0:
            dist.barrier()
            WRITER.add_scalars(main_tag='LOSS--TRAIN',
                               tag_scalar_dict={
                                   'TOTAL': total_loss / datalen,
                                   # 'p2':total_loss_p2 / datalen,
                                   'p5':total_loss_p5 / datalen,
                                   # 'p18':total_loss_p18 / datalen
                               },
                               global_step=epoch_idx + 1)
            for name, param in net.named_parameters():
                WRITER.add_histogram(name, param.clone().cpu().data.numpy(), epoch_idx + 1)
                WRITER.add_histogram(name + '/grad', param.grad.clone().cpu().data.numpy(), epoch_idx + 1)

            if validation:
                val_result = validate(net, val_dataset, [lossfn_p5], local_rank)
                total_loss_val, p5_loss_val = val_result['total_loss'], val_result['p5']['loss']
                # total_loss_val, p2_loss_val, p5_loss_val, p18_loss_val = \
                #     val_result['total_loss'], val_result['p2']['loss'], val_result['p5']['loss'], val_result['p18'][
                #         'loss']
                WRITER.add_scalars(main_tag='LOSS--EVAL',
                                   tag_scalar_dict={
                                       'TOTAL': total_loss_val,
                                       # 'p2': p2_loss_val,
                                       'p5': p5_loss_val,
                                       # 'p18': p18_loss_val
                                   },
                                   global_step=epoch_idx + 1)

                p5_recall_val = val_result['p5']['recall']
                p5_precision_val = val_result['p5']['precision']
                # # p2 的auc/roc曲线
                # WRITER.add_scalars(main_tag='AUC/ROC--DOWN_p2',
                #                    tag_scalar_dict={
                #                        'recall': p2_recall_val[1],
                #                        'precision': p2_precision_val[1]
                #                    },
                #                    global_step=epoch_idx + 1)
                # WRITER.add_scalars(main_tag='AUC/ROC--UP_p2',
                #                    tag_scalar_dict={
                #                        'recall': p2_recall_val[2],
                #                        'precision': p2_precision_val[2]
                #                    },
                #                    global_step=epoch_idx + 1)
                # p5 的auc/roc曲线
                WRITER.add_scalars(main_tag='AUC/ROC--DOWN_p5',
                                   tag_scalar_dict={
                                       'recall': p5_recall_val[1],
                                       'precision': p5_precision_val[1]
                                   },
                                   global_step=epoch_idx + 1)
                WRITER.add_scalars(main_tag='AUC/ROC--UP_p5',
                                   tag_scalar_dict={
                                       'recall': p5_recall_val[2],
                                       'precision': p5_precision_val[2]
                                   },
                                   global_step=epoch_idx + 1)
                # p18 的auc/roc曲线
                # WRITER.add_scalars(main_tag='AUC/ROC--DOWN_p18',
                #                    tag_scalar_dict={
                #                        'recall': p18_recall_val[1],
                #                        'precision': p18_precision_val[1]
                #                    },
                #                    global_step=epoch_idx + 1)
                # WRITER.add_scalars(main_tag='AUC/ROC--UP_p18',
                #                    tag_scalar_dict={
                #                        'recall': p18_recall_val[2],
                #                        'precision': p18_precision_val[2]
                #                    },
                #                    global_step=epoch_idx + 1)
                #
                p5_accuracy_val = val_result['p5']['accuracy']
                WRITER.add_scalars(main_tag='ACCURACY',
                                   tag_scalar_dict={
                                       # 'p2': p2_accuracy_val,
                                       'p5': p5_accuracy_val,
                                       # 'p18': p18_accuracy_val
                                   },
                                   global_step=epoch_idx + 1)

                LOGGER.info(f"Epoch {epoch_idx} validation loss: {p5_loss_val}.")
            torch.save({
                'epoch': epoch_idx + 1,
                'train_loss': total_loss / datalen,
                'validation_result': val_result,
                'state_dict': ddp_net.state_dict(),
                'optimizer_dict': opt.optimiser.state_dict(),
            }, os.path.join(model_path, f'CNNLstmCLS_epoch_{epoch_idx}_bs{BATCH_SIZE}_sl{SEQ_LEN}_ts{TIMESTEP}.pth.tar'))
            LOGGER.info(f'Epoch {epoch_idx} finished.')

        else:
            dist.barrier()

        del loss
        gc.collect()
        torch.cuda.empty_cache()

        LOGGER.info(f"Epoch {epoch_idx} is done.\n")

    LOGGER.info("Training has finished.")

    return

class MetricRUC:
    def __init__(self, name: str, classnum: int):
        self.name = name.upper()

        self.correct = 0
        self.total = 0
        self.target_num = torch.zeros((1, classnum))
        self.predict_num = torch.zeros((1, classnum))
        self.acc_num = torch.zeros((1, classnum))

    def update(self, netout, target):

        target = target.detach().cpu().long()
        netout = netout.detach().cpu()
        _, pred = torch.max(netout.data, 1)
        self.total += target.size(0)
        self.correct += pred.eq(target.data).cpu().sum()
        pre_mask = torch.zeros(netout.size()).scatter_(1, pred.view(-1, 1), 1.)
        self.predict_num += pre_mask.sum(0)
        tar_mask = torch.zeros(netout.size()).scatter_(1, target.view(-1, 1), 1.)
        self.target_num += tar_mask.sum(0)
        acc_mask = pre_mask * tar_mask
        self.acc_num += acc_mask.sum(0)

    def summary(self):
        recall = self.acc_num / self.target_num
        precision = self.acc_num / self.predict_num
        F1 = 2 * recall * precision / (recall + precision)
        accuracy = self.acc_num.sum(1) / self.target_num.sum(1)
        # 精度调整
        recall = (recall.numpy()[0] * 100).round(3)
        precision = (precision.numpy()[0] * 100).round(3)
        F1 = (F1.numpy()[0] * 100).round(3)
        accuracy = (accuracy.numpy()[0] * 100).round(3)

        print(f'-----------------------{self.name}-------------------------')
        print(f'{self.name} RECALL', " ".join('%s' % id for id in recall))
        print(f'{self.name} PRECISION', " ".join('%s' % id for id in precision))
        print(f'{self.name} F1', " ".join('%s' % id for id in F1))
        print(f'{self.name} accuracy', accuracy)

        return recall, precision, accuracy, F1




def validate(net, validation_dataset, lossfn, local_rank):

    net.eval()

    result_dict = {}

    # total_loss_p2 = 0.0
    total_loss_p5 = 0.0
    # total_loss_p18 = 0.0

    # summary_p2 = MetricRUC(name = 'p2', classnum=OUTPUT_SIZE//3)
    summary_p5 = MetricRUC(name = 'p5', classnum=OUTPUT_SIZE//3)
    # summary_p18 = MetricRUC(name = 'p18', classnum=OUTPUT_SIZE//3)
    total_loss = 0
    with torch.no_grad():
        for x, y in validation_dataset:
            y_p5 = y

            # y_p18 = y[..., 2]
            h_p5 = net(x.permute(0, 2, 1).to(local_rank))

            # loss_p2 = lossfn[0](h_p2, y_p2.to(local_rank).long())
            loss_p5 = lossfn[0](h_p5, y_p5.to(local_rank).long())
            # loss_p18 = lossfn[0](h_p18, y_p18.to(local_rank).long())

            total_loss += (loss_p5).item()

            # total_loss_p2 += loss_p2.item()
            total_loss_p5 += loss_p5.item()
            # total_loss_p18 += loss_p18.item()

            # summary_p2.update(h_p2, y_p2)
            summary_p5.update(h_p5, y_p5)
            # summary_p18.update(h_p18, y_p18)

        # recall, precision, accuracy, F1 = summary_p2.summary()
        result_dict['total_loss'] = total_loss / len(validation_dataset)
        # result_dict['p2'] = {'loss': loss_p2 / len(validation_dataset),
        #                      'recall': recall,
        #                      'precision': precision,
        #                      'accuracy': accuracy,
        #                      'F1': F1}
        recall, precision, accuracy, F1 = summary_p5.summary()
        result_dict['p5'] = {'loss': loss_p5 / len(validation_dataset),
                             'recall': recall,
                             'precision': precision,
                             'accuracy': accuracy,
                             'F1': F1}
        # recall, precision, accuracy, F1 = summary_p18.summary()
        # result_dict['p18'] = {'loss': loss_p18 / len(validation_dataset),
        #                       'recall': recall,
        #                       'precision': precision,
        #                       'accuracy': accuracy,
        #                       'F1': F1}
        return result_dict



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

    # world_size = 1
    mp.spawn(train,
             args=(world_size, world_dict, shard_dict, True, None, ),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    main_train(Logger)