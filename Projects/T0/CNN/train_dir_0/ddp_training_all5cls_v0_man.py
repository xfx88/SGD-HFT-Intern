"""
5分类模型
"""
import sys
sys.path.append("/home/yby/SGD-HFT-Intern/Projects/T0/CNN")

import torch.multiprocessing as mp
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import utilities as ut
from torch.nn.utils import weight_norm
from torch.optim import lr_scheduler, SGD


import math
import random
import gc
import src.logger as logger
from src.dataset_generator import DatasetAll, DatasetGenerator
from utilities import *

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12309"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,5,6,7,8,9"
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

INPUT_SIZE = 37
OUTPUT_SIZE = 15
SEQ_LEN = 64
TIMESTEP = 2

EPOCHS = 80

validation_dates = ['20211020','20211021','20211022','20211025','20211026','20211027','20211028','20211029']

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MatrixCrossEntropy(nn.Module):
    def __init__(self, label_nums = 3, local_rank = 0, gamma = 2):
        class_counts = torch.Tensor([[955, 205, 32, 198, 33],
                                     [792, 286, 36, 270, 39],
                                     [580, 383, 52, 347, 60]])

        class_weight = 1 / class_counts
        super(MatrixCrossEntropy, self).__init__()
        assert len(class_weight.shape) == 2
        # self.weight = F.normalize(1 / torch.Tensor([[25,1,1],[19,1,1],[14,1,1]]), dim = 1, p = 1).to(local_rank)
        # self.class_weight = F.normalize(1 / torch.Tensor([[1e5,1,1],[1e5,1,1],[1e5,1,1]]), dim = 1, p = 1).to(local_rank)
        self.class_weight = F.normalize(class_weight, dim = 1, p = 1).to(local_rank)
        # print(self.class_weight)
        self.label_weight = torch.Tensor([1/3, 1/3, 1/3]).to(local_rank)
        # self.base_loss = [nn.CrossEntropyLoss(weight=self.class_weight[i], reduction="mean") for i in range(label_nums)]

        self.label_nums = list(range(label_nums))
        self.gamma = gamma

    def forward(self, pred, target):
        temp = []
        for i in self.label_nums:
            temp.append(F.one_hot(target[:, i].unsqueeze(1), num_classes = self.class_weight.shape[-1]))
        temp = torch.cat(temp, dim = 1)
        # logsoftmax_pred = F.log_softmax(pred, dim=2)
        # softmax_pred = F.softmax(pred, dim=2)
        # loss = 0
        # for i in self.label_nums:
        #     loss += self.base_loss[i](pred[:, i, :], target[:, i]) * self.label_weight[i]
        # temp:(N, 3, 3), class_weight:(3,3), pred:(N, 3, 3) >> (N, 3, 3) >>sum >> (N, 3)
        loss = - (((temp * self.class_weight * F.log_softmax(pred, dim=2)).sum(2)) @ self.label_weight).mean()

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

        # self.batchNorm = nn.BatchNorm1d(num_features = out_channels)

        self.cnn1 = weight_norm(nn.Conv1d(in_channels=in_channels,
                                          out_channels=_latent_size,
                                          kernel_size=(kernel_size,),
                                          padding=(_padding_size,),
                                          dilation=(dilation_size,)))

        self.cnn2 = weight_norm(nn.Conv1d(_latent_size,
                                          out_channels=out_channels,
                                          kernel_size=(kernel_size,),
                                          padding=(_padding_size,),
                                          dilation=(dilation_size,)))

        self.sub_block = nn.Sequential(self.cnn1, nn.ReLU(), nn.Dropout(0.2), self.cnn2, nn.ReLU(), nn.Dropout(0.2))
        self.output_elu = nn.ReLU()
        self.resample = nn.Conv1d(in_channels,
                                  out_channels,
                                  kernel_size=(1,),
                                  padding=(0,)) \
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
        x = self.output_elu(x + res)

        return x


class ConvLstmNet(nn.Module):
    def __init__(self, in_features, seq_len, out_features):
        super(ConvLstmNet, self).__init__()

        # set size
        self.in_features = in_features
        self.seq_len     = seq_len
        self.out_features = out_features

        self.bn = nn.BatchNorm1d(in_features)

        _kernel_size = 3

        _layers = []
        # _levels = [INPUT_SIZE, 64, 96, 128, 96, 64, 32]
        _levels = [INPUT_SIZE] * 9
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

        x = self.bn(x)
        x = self.network(x)
        x = self.featureExtractor(x.flatten(1)).view((-1, 3, OUTPUT_SIZE // 3))
        # x = self.sigmoid(x)

        return x


def train(local_rank, world_size, start_date, end_date, validation = False, Resume = None):
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        tsboard_path = "/home/yby/SGD-HFT-Intern/Projects/T0/CNN/tensorboard_logs/cls5all_relu_v2_manu"
        if not os.path.exists(tsboard_path):
            os.makedirs(tsboard_path)
        WRITER = SummaryWriter(log_dir=tsboard_path)

    model_path = '/home/yby/SGD-HFT-Intern/Projects/T0/CNN/train_dir_0/model/CNN_param_cls5all_relu_v2_manu'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if Resume:
        model_name = "CNNLstmCLS_epoch_25_bs10000_sl64_ts2.pth.tar"
        model_data = torch.load(os.path.join(model_path, model_name), map_location='cpu')
        model_data['state_dict'] = ddpModel_to_normal(model_data['state_dict'])

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
    optimizer = torch.optim.Adam(ddp_net.parameters(), lr = 1e-3)
    # optimizer = SGD(ddp_net.parameters(), lr = 1e-3, momentum = 0.5)

    # if prior_epochs != 0:
    #     optimizer.load_state_dict(model_data['optimizer_dict'])
        # scheduler.load_state_dict((model_data['scheduler_dict']))
    scheduler = lr_scheduler.StepLR(optimizer, step_size= 5, gamma=0.85)
    ds_gen = DatasetGenerator(start_date=start_date,
                              end_date = end_date,
                              local_rank = local_rank,
                              world_size = world_size,
                              batch_size = BATCH_SIZE,
                              )

    ddp_net.train()
    loss_fn = MatrixCrossEntropy(local_rank=local_rank)

    LOGGER.info(f'local rank {local_rank}: Dataset is prepared.')
    LOGGER.info(f'local rank {local_rank}: GPU calculating.')
    for epoch_idx in range(prior_epochs, EPOCHS):
        ddp_net.train()
        train_dataset, val_dataset = ds_gen.update(epoch_idx)

        loss = 0
        total_loss = 0.0
        datalen = 0
        for batch_idx, (x, y) in enumerate(train_dataset):
            datalen += 1
            optimizer.zero_grad()
            # h_p2, h_p5, h_p18 = net(x.permute(0,2,1).to(local_rank))
            h = net(x.permute(0, 2, 1).to(local_rank))

            loss = loss_fn(h, y.to(local_rank))

            total_loss += loss.item()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(ddp_net.parameters(), max_norm=1e4)
            optimizer.step()


            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                LOGGER.info(f"Epoch {epoch_idx}: {batch_idx + 1}), loss {loss.item()},")

        if not (epoch_idx >= 20 and epoch_idx < 50):
            scheduler.step()

        if local_rank == 0:
            dist.barrier()
            WRITER.add_scalars(main_tag='LOSS--TRAIN',
                               tag_scalar_dict={
                                   'TOTAL': total_loss / datalen,
                               },
                               global_step=epoch_idx + 1)
            # for name, param in ddp_net.named_parameters():
            #     WRITER.add_histogram(name, param.clone().cpu().data.numpy(), epoch_idx + 1)
            #     WRITER.add_histogram(name + '/grad', param.grad.clone().cpu().data.numpy(), epoch_idx + 1)

            if validation:
                # val_result = validate(net, val_dataset, [lossfn_p2, lossfn_p5, lossfn_p18], local_rank)
                val_result = validate(net, val_dataset, loss_fn, local_rank)
                total_loss_val = val_result['total_loss']
                WRITER.add_scalars(main_tag='LOSS--EVAL',
                                   tag_scalar_dict={
                                       'TOTAL': total_loss_val,
                                   },
                                   global_step=epoch_idx + 1)

                p2_accuracy_val, p5_accuracy_val, p18_accuracy_val = \
                    val_result['p2']['accuracy'], val_result['p5']['accuracy'], val_result['p18']['accuracy']
                WRITER.add_scalars(main_tag='ACCURACY',
                                   tag_scalar_dict={
                                       'p2': p2_accuracy_val,
                                       'p5': p5_accuracy_val,
                                       'p18': p18_accuracy_val,
                                       # 'diff': diff_accuracy_val
                                   },
                                   global_step=epoch_idx + 1)

                LOGGER.info(f"Epoch {epoch_idx} validation loss: {total_loss_val}.")
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

        del loss, train_dataset, val_dataset
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
    # total_loss_p5 = 0.0
    # total_loss_p18 = 0.0

    summary_p2 = MetricRUC(name = 'p2', classnum=OUTPUT_SIZE//3)
    summary_p5 = MetricRUC(name = 'p5', classnum=OUTPUT_SIZE//3)
    summary_p18 = MetricRUC(name = 'p18', classnum=OUTPUT_SIZE//3)
    # summary_diff = MetricRUC(name = 'diff', classnum=OUTPUT_SIZE//4)
    total_loss = 0
    datalen = 0
    with torch.no_grad():
        for x, y in validation_dataset:
            datalen += 1

            h = net(x.permute(0, 2, 1).to(local_rank))

            h_p2 = h[:, 0, :].squeeze(1)
            h_p5 = h[:, 1, :].squeeze(1)
            h_p18 = h[:, 2, :].squeeze(1)
            # h_diff = h[:, 3, :].squeeze(1)

            loss = lossfn(h, y.to(local_rank))
            total_loss += loss.item()

            y_p2 = y[..., 0]
            y_p5 = y[..., 1]
            y_p18 = y[..., 2]
            # y_diff = y[..., 3]
            # loss_p2 = lossfn[0](h_p2, y_p2.to(local_rank).long())
            # loss_p5 = lossfn[1](h_p5, y_p5.to(local_rank).long())
            # loss_p18 = lossfn[2](h_p18, y_p18.to(local_rank).long())

            # total_loss_p2 += loss_p2.item()
            # total_loss_p5 += loss_p5.item()
            # total_loss_p18 += loss_p18.item()
            # total_loss += (1/9 * loss_p2 + 3/9 * loss_p5 + 5/9 * loss_p18).item()

            summary_p2.update(h_p2, y_p2)
            summary_p5.update(h_p5, y_p5)
            summary_p18.update(h_p18, y_p18)
            # summary_diff.update(h_diff, y_diff)

        recall, precision, accuracy, F1 = summary_p2.summary()
        result_dict['total_loss'] = total_loss / datalen
        result_dict['p2'] = {'recall': recall,
                             # 'loss': loss_p2 / len(validation_dataset),
                             'precision': precision,
                             'accuracy': accuracy,
                             'F1': F1}
        recall, precision, accuracy, F1 = summary_p5.summary()
        result_dict['p5'] = {'recall': recall,
                             # 'loss': loss_p5 / len(validation_dataset),
                             'precision': precision,
                             'accuracy': accuracy,
                             'F1': F1}
        recall, precision, accuracy, F1 = summary_p18.summary()
        result_dict['p18'] = {'recall': recall,
                              # 'loss': loss_p18 / len(validation_dataset),
                              'precision': precision,
                              'accuracy': accuracy,
                              'F1': F1}
        # recall, precision, accuracy, F1 = summary_diff.summary()
        # result_dict['diff'] = {'recall': recall,
        #                       # 'loss': loss_p18 / len(validation_dataset),
        #                       'precision': precision,
        #                       'accuracy': accuracy,
        #                       'F1': F1}
        return result_dict



def main_train(LOGGER):

    train_start_date = '20210701'
    train_end_date = '20211031'
    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    LOGGER.info('Splitting indices for distributed training.')

    # world_size = 1
    mp.spawn(train,
             args=(world_size, train_start_date, train_end_date, True, None),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    main_train(Logger)