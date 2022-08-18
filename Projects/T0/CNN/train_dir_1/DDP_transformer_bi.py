import numpy as np
import os
import gc
import random
import pickle
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from collections import OrderedDict
import math

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import utilities as ut
from utilities import factor_ret_cols
from src.dataset import HFDataset, HFDatasetVal
import src.logger as logger
from tst import TransformerCLS


GLOBAL_SEED = 12309
DB_ID = 0
WARMUP = 500
BATCH_SIZE = 1
SEQ_LEN = 64
INPUT_SIZE = 44
OUTPUT_SIZE = 6
EPOCHS = 80

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12359"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9"
os.environ["OMP_NUM_THREADS"] = "8"
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

Logger = logger.getLogger()
Logger.setLevel("INFO")


class Optimiser:

    def __init__(self, model, optimiser=None, scale_factor=1.0, warmup_steps=2000, beta1=0.9, beta2=0.98, epsilon=1e-9):

        if optimiser is not None: self.optimiser = optimiser
        # else: self.optimiser = torch.optim.Adam(model.parameters(), lr=0, betas=(beta1, beta2), eps=epsilon)
        else: self.optimiser = torch.optim.SGD(model.parameters(), lr = 0)
        self.scale_factor = scale_factor
        self.warmup_steps = math.pow(warmup_steps, -1.5)
        self.current_step = 0
        self.inv_sqrt_d_input = math.pow(model.module.d_input, -0.5)

        self.lrate = lambda step: self.inv_sqrt_d_input * min(math.pow(step, -0.5), step * self.warmup_steps)
        self.rate = None

    def step(self):
        self.current_step += 1
        self.rate = self.scale_factor * self.lrate(self.current_step)
        for i in self.optimiser.param_groups: i['lr'] = self.rate
        self.optimiser.step()

    def zero_grad(self):
        self.optimiser.zero_grad()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=5, num_classes = 3, size_average=False):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            # print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            # print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = torch.sigmoid(preds)
        preds_logsoft = torch.log(preds_softmax)
        # preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        # preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def shard_keys(start_date, end_date, seq_len = 50, time_step = 1, db = DB_ID):
    shard_dict_whole = dict()
    rs = ut.redis_connection(db=db)
    all_redis_keys = rs.keys()
    keys_to_shard = [x.decode(encoding = 'utf-8') for x in all_redis_keys
                    if ((len(x.decode(encoding = 'utf-8').split('_')) == 3)
                    and (x.decode(encoding = 'utf-8').split('_')[2] <= end_date[4:6])
                    and (x.decode(encoding = 'utf-8').split('_')[2] >= start_date[4:6])
                    and (x.decode(encoding = 'utf-8').split('_')[0] == 'bilabels'))]
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
    # random.seed(GLOBAL_SEED)
    len_part = math.ceil(key_num / world_size)
    random.Random(GLOBAL_SEED).shuffle(idx_list)
    world_dict = {i: [idx_list[j] for j in idx_list[i * len_part: (i + 1) * len_part]]
                  for i in range(world_size)}

    return world_dict, shard_dict

# param5 跨样本单特征标准化
tOpt = ut.TrainingOptions(BATCH_SIZE=BATCH_SIZE,
                          EPOCHS=EPOCHS,
                          N_stack=2,
                          heads=2,
                          query=16,
                          value=16,
                          d_model=128,
                          d_input=INPUT_SIZE,
                          d_output=OUTPUT_SIZE,
                          chunk_mode = None
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
        tsboard_path = "/home/wuzhihan/Projects/CNN/tensorboard_logs/transformer/BICLS"
        if not os.path.exists(tsboard_path):
            os.makedirs(tsboard_path)
        WRITER = SummaryWriter(log_dir=tsboard_path)

    model_path = '/home/wuzhihan/Projects/CNN/train_dir_1/transformer/param_bicls/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if Resume:
        model_name = f'TST_epoch_3_bs5000_sl64.pth.tar'
        model_data = torch.load(os.path.join(model_path, model_name), map_location='cpu')
        model_data['state_dict'] = ddpModel_to_normal(model_data['state_dict'])

    LOGGER = logger.getLogger()
    LOGGER.setLevel('INFO') if local_rank == 0 else LOGGER.setLevel('WARNING')
    prior_epochs = 0 if not Resume else model_data['epoch'] + 1

    net = TransformerCLS(tOpt).to(local_rank)

    if prior_epochs != 0:
        LOGGER.info(f"Prior epoch: {prior_epochs}, training resumes.")
        net.load_state_dict(model_data['state_dict'])
    else:
        LOGGER.info(f"No prior epoch, training start.")

    ddp_net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    # lossfn_p2 = FocalLoss(alpha = 0.8, gamma = 2).to(local_rank)
    # lossfn_p5 = FocalLoss(alpha = 0.75, gamma = 2).to(local_rank)
    # lossfn_p18 = FocalLoss(alpha = 0.7, gamma = 2).to(local_rank)
    lossfn_p2 = nn.CrossEntropyLoss(weight = torch.tensor([0.2, 0.8])).to(local_rank)
    lossfn_p5 = nn.CrossEntropyLoss(weight = torch.tensor([0.25, 0.75])).to(local_rank)
    lossfn_p18 = nn.CrossEntropyLoss(weight = torch.tensor([0.3, 0.7])).to(local_rank)

    opt = Optimiser(ddp_net, scale_factor=1e-4, warmup_steps=WARMUP)
    if prior_epochs != 0:
        opt.optimiser.load_state_dict(model_data['optimizer_dict'])

    ddp_net.train()
    local_rank_id = world_dict[local_rank]
    local_rank_id.sort()
    len_train = int(len(local_rank_id) * 0.75)
    train_ids = local_rank_id[:len_train]
    val_ids = local_rank_id[len_train:]

    train_dataset = HFDataset(local_ids=train_ids,
                              shard_dict=shard_dict,
                              batch_size=BATCH_SIZE,
                              seq_len=SEQ_LEN)

    val_dataset = HFDatasetVal(local_ids=val_ids,
                               shard_dict=shard_dict,
                               batch_size=BATCH_SIZE,
                               seq_len=SEQ_LEN)


    LOGGER.info(f'local rank {local_rank}: Dataset is prepared.')
    LOGGER.info(f'local rank {local_rank}: GPU calculating.')
    datalen = len(train_dataset)
    for epoch_idx in range(prior_epochs, EPOCHS):
        ddp_net.train()

        total_loss_p2 = 0.0
        total_loss_p5 = 0.0
        total_loss_p18 = 0.0
        total_loss = 0.0
        DENOMINATOR = 2
        epoch_denominator = 3
        if epoch_idx != 0 and epoch_idx % epoch_denominator == 0:
            opt.scale_factor /= DENOMINATOR
            DENOMINATOR -= 0.02

        for batch_idx, (x, y) in enumerate(train_dataset):
            opt.zero_grad()
            h_p2, h_p5, h_p18 = net(x.to(local_rank))

            loss_p2 = lossfn_p2(h_p2.permute(2,0,1).flatten(1).T, y[..., 0].flatten().cuda().long())
            loss_p5 = lossfn_p5(h_p5.permute(2,0,1).flatten(1).T, y[..., 1].flatten().cuda().long())
            loss_p18 = lossfn_p18(h_p18.permute(2,0,1).flatten(1).T, y[..., 2].flatten().cuda().long())
            total_loss_p2 += loss_p2.item()
            total_loss_p5 += loss_p5.item()
            total_loss_p18 += loss_p18.item()

            loss = 1 / 9 * loss_p2 + 3 / 9 * loss_p5 + 5 / 9 * loss_p18

            total_loss += loss.item()
            loss.backward()
            opt.step()

            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                LOGGER.info(f"Epoch {epoch_idx}: {batch_idx + 1} / {datalen}), loss {loss.item()},")

        del x, y
        gc.collect()

        if local_rank == 0:
            dist.barrier()
            WRITER.add_scalars(main_tag='LOSS--TRAIN',
                               tag_scalar_dict={
                                   'TOTAL': total_loss / datalen,
                                   'p2': total_loss_p2 / datalen,
                                   'p5': total_loss_p5 / datalen,
                                   'p18': total_loss_p18 / datalen
                               },
                               global_step=epoch_idx + 1)
            for name, param in ddp_net.named_parameters():
                WRITER.add_histogram(name, param.clone().cpu().data.numpy(), epoch_idx + 1)
                WRITER.add_histogram(name + '/grad', param.grad.clone().cpu().data.numpy(), epoch_idx + 1)

            if validation:
                val_result = validate(net, val_dataset, [lossfn_p2, lossfn_p5, lossfn_p18], local_rank)
                total_loss_val, p2_loss_val, p5_loss_val, p18_loss_val = \
                    val_result['total_loss'], val_result['p2']['loss'], val_result['p5']['loss'], val_result['p18'][
                        'loss']
                WRITER.add_scalars(main_tag='LOSS--EVAL',
                                   tag_scalar_dict={
                                       'TOTAL': total_loss_val,
                                       'p2': p2_loss_val,
                                       'p5': p5_loss_val,
                                       'p18': p18_loss_val
                                   },
                                   global_step=epoch_idx + 1)

                p2_recall_val, p5_recall_val, p18_recall_val = \
                    val_result['p2']['recall'], val_result['p5']['recall'], val_result['p18']['recall']
                p2_precision_val, p5_precision_val, p18_precision_val = \
                    val_result['p2']['precision'], val_result['p5']['precision'], val_result['p18']['precision']
                # # p2 的auc/roc曲线
                WRITER.add_scalars(main_tag='AUC/ROC--DOWN_p2',
                                   tag_scalar_dict={
                                       'recall': p2_recall_val[0],
                                       'precision': p2_precision_val[0]
                                   },
                                   global_step=epoch_idx + 1)
                WRITER.add_scalars(main_tag='AUC/ROC--UP_p2',
                                   tag_scalar_dict={
                                       'recall': p2_recall_val[1],
                                       'precision': p2_precision_val[1]
                                   },
                                   global_step=epoch_idx + 1)
                # p5 的auc/roc曲线
                WRITER.add_scalars(main_tag='AUC/ROC--DOWN_p5',
                                   tag_scalar_dict={
                                       'recall': p5_recall_val[0],
                                       'precision': p5_precision_val[0]
                                   },
                                   global_step=epoch_idx + 1)
                WRITER.add_scalars(main_tag='AUC/ROC--UP_p5',
                                   tag_scalar_dict={
                                       'recall': p5_recall_val[1],
                                       'precision': p5_precision_val[1]
                                   },
                                   global_step=epoch_idx + 1)
                # p18 的auc/roc曲线
                WRITER.add_scalars(main_tag='AUC/ROC--DOWN_p18',
                                   tag_scalar_dict={
                                       'recall': p18_recall_val[0],
                                       'precision': p18_precision_val[0]
                                   },
                                   global_step=epoch_idx + 1)
                WRITER.add_scalars(main_tag='AUC/ROC--UP_p18',
                                   tag_scalar_dict={
                                       'recall': p18_recall_val[0],
                                       'precision': p18_precision_val[1]
                                   },
                                   global_step=epoch_idx + 1)
                #
                p2_accuracy_val, p5_accuracy_val, p18_accuracy_val = \
                    val_result['p2']['accuracy'], val_result['p5']['accuracy'], val_result['p18']['accuracy']
                WRITER.add_scalars(main_tag='ACCURACY',
                                   tag_scalar_dict={
                                       'p2': p2_accuracy_val,
                                       'p5': p5_accuracy_val,
                                       'p18': p18_accuracy_val
                                   },
                                   global_step=epoch_idx + 1)

                LOGGER.info(f"Epoch {epoch_idx} validation loss: {p18_loss_val}.")
            torch.save({
                'epoch': epoch_idx + 1,
                'train_loss': total_loss / datalen,
                'validation_result': val_result,
                'state_dict': ddp_net.state_dict(),
                'optimizer_dict': opt.optimiser.state_dict(),
            }, os.path.join(model_path,
                            f'TST_epoch_{epoch_idx}_bs{BATCH_SIZE}_sl{SEQ_LEN}.pth.tar'))
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

        del target, netout
        gc.collect()

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

    total_loss_p2 = 0.0
    total_loss_p5 = 0.0
    total_loss_p18 = 0.0

    summary_p2 = MetricRUC(name = 'p2', classnum=OUTPUT_SIZE//3)
    summary_p5 = MetricRUC(name = 'p5', classnum=OUTPUT_SIZE//3)
    summary_p18 = MetricRUC(name = 'p18', classnum=OUTPUT_SIZE//3)
    total_loss = 0
    with torch.no_grad():
        for x, y in validation_dataset:

            y_p2 = y[..., 0]
            y_p5 = y[..., 1]
            y_p18 = y[..., 2]
            h_p2, h_p5, h_p18 = net(x.to(local_rank))


            y_p2, y_p5, y_p18 = y_p2.flatten(), y_p5.flatten(), y_p18.flatten()

            loss_p2 = lossfn[0](h_p2.permute(2,0,1).flatten(1).T, y_p2.flatten().to(local_rank).long())
            loss_p5 = lossfn[1](h_p5.permute(2,0,1).flatten(1).T, y_p5.flatten().to(local_rank).long())
            loss_p18 = lossfn[2](h_p18.permute(2,0,1).flatten(1).T, y_p18.flatten().to(local_rank).long())

            total_loss_p2 += loss_p2.item()
            total_loss_p5 += loss_p5.item()
            total_loss_p18 += loss_p18.item()
            total_loss += (1/3 * loss_p2 + 1/3 * loss_p5 + 1/3 * loss_p18).item()

            h_p2 = h_p2.transpose(0, 2).flatten(1).T
            h_p5 = h_p5.transpose(0, 2).flatten(1).T
            h_p18 = h_p18.transpose(0, 2).flatten(1).T

            summary_p2.update(h_p2, y_p2)
            summary_p5.update(h_p5, y_p5)
            summary_p18.update(h_p18, y_p18)

        recall, precision, accuracy, F1 = summary_p2.summary()
        result_dict['total_loss'] = total_loss / len(validation_dataset)
        result_dict['p2'] = {'loss': loss_p2 / len(validation_dataset),
                             'recall': recall,
                             'precision': precision,
                             'accuracy': accuracy,
                             'F1': F1}
        recall, precision, accuracy, F1 = summary_p5.summary()
        result_dict['p5'] = {'loss': loss_p5 / len(validation_dataset),
                             'recall': recall,
                             'precision': precision,
                             'accuracy': accuracy,
                             'F1': F1}
        recall, precision, accuracy, F1 = summary_p18.summary()
        result_dict['p18'] = {'loss': loss_p18 / len(validation_dataset),
                              'recall': recall,
                              'precision': precision,
                              'accuracy': accuracy,
                              'F1': F1}
        return result_dict


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