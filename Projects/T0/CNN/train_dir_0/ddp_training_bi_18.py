import sys
sys.path.append("/home/yby/SGD-HFT-Intern/Projects/T0/CNN")

import numpy as np
import math
import random
import gc
import os

from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import weight_norm
from torch.optim import lr_scheduler, SGD

import src.logger as logger
from src.dataset_clsall import HFDatasetBi
import utilities as ut

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12310"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9"
os.environ["OMP_NUM_THREADS"] = '2'

DB_ID = 0

Logger = logger.getLogger()
Logger.setLevel("INFO")


GLOBAL_SEED = 12310
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

BATCH_SIZE = 10000
WARMUP = 500
RESUME = None

INPUT_SIZE = 44
OUTPUT_SIZE = 6
SEQ_LEN = 64
TIMESTEP = 3

EPOCHS = 80

validation_dates = ['20211020','20211021','20211022','20211025','20211026','20211027','20211028','20211029']

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=False):
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
        # preds_softmax = torch.sigmoid(preds)
        # preds_logsoft = torch.log(preds_softmax)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

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

class MatrixCrossEntropy(nn.Module):
    def __init__(self, label_nums = 3, local_rank = 0, gamma = 0, class_num = 2):
        super(MatrixCrossEntropy, self).__init__()
        # self.weight = F.normalize(1 / torch.Tensor([[25,1,1],[19,1,1],[14,1,1]]), dim = 1, p = 1).to(local_rank)
        self.weight = F.normalize(1 / torch.Tensor([[2492,411],[2330,624],[2120,834]]), dim = 1, p = 1).to(local_rank)
        self.label_nums = list(range(label_nums))
        self.gamma = gamma
        self.class_num = class_num

    def forward(self, pred, target):
        temp = []
        for i in self.label_nums:
            temp.append(F.one_hot(target[:, i].unsqueeze(1), num_classes = self.class_num))
        temp = torch.cat(temp, dim = 1)
        logsoftmax_pred = F.log_softmax(pred, dim=2)
        softmax_pred = F.softmax(pred, dim=2)
        loss = - (torch.pow((1 - softmax_pred), self.gamma) * temp * self.weight * logsoftmax_pred).sum()

        return loss

class CNNBlock(nn.Module):
    def __init__(self, in_channels: int,
                       out_channels: int,
                       kernel_size: tuple,
                       padding: tuple,):

        super(CNNBlock, self).__init__()
        self.conv = weight_norm(nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding = padding))
        # self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x):

        x = self.conv(x)
        # x = F.relu(x)

        return x

class OuputLayer(nn.Module):
    def __init__(self, in_features, seq_len, out_features):

        super(OuputLayer, self).__init__()

        # self.bn_res0 = nn.BatchNorm1d(in_features)
        # self.bn_res1 = nn.BatchNorm1d(64)
        # self.bn_res2 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(0.3)

        self.downsample0 = nn.Conv1d(in_channels=256,out_channels=128,dilation=(8,),kernel_size=(1,),padding=(0,))
        self.downsample1 = nn.Conv1d(in_channels=128,out_channels=64,dilation=(6,),kernel_size=(1,),padding=(0,))
        self.downsample2 = nn.Conv1d(in_channels=64,out_channels=in_features,dilation=(4,),kernel_size=(1,),padding=(0,))

        self.out = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=(seq_len, ))

        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight.data, nonlinearity='conv1d')
                nn.init.zeros_(layer.bias.data)
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight.data, nonlinearity='linear')
                nn.init.zeros_(layer.bias.data)

    def forward(self, x, res_0, res_1, res_2):
        # res_0 N, in_feature, 64
        # res_1 N, 128, 64
        # res_2 N, 256, 64

        x = self.downsample0(x) + res_2
        x = self.downsample1(x) + res_1
        x = self.downsample2(x) + res_0
        x = self.dropout(x)

        x = self.out(x).squeeze(-1)

        return x

class ConvLstmNet(nn.Module):
    def __init__(self, in_features, seq_len, out_features):
        super(ConvLstmNet, self).__init__()
        # set size
        self.in_features = in_features
        self.seq_len     = seq_len
        self.out_features = out_features

        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm1d(in_features)

        # self.decomp = nn.Conv1d(in_channels=in_features, out_channels=self.out_features, kernel_size=(14,), dilation=(4,), stride=(4,))

        self.conv_0 =  CNNBlock(in_channels= in_features,out_channels = 64,kernel_size=(3, ),padding=(1, ))
        self.conv_1 =  CNNBlock(in_channels= 64,out_channels = 128,kernel_size=(3, ),padding=(1, ))
        self.conv_2 = CNNBlock(in_channels=128,out_channels=256,kernel_size=(3,),padding=(1,))

        self.seqExtractor1 = nn.Linear(seq_len, 24)

        self.conv_3 = CNNBlock(in_channels=256,out_channels=512,kernel_size=(3,),padding=(1,))

        self.conv_4 = CNNBlock(in_channels=512,out_channels=128,kernel_size=(3,),padding=(1,))

        self.seqExtractor2 = nn.Linear(24, 3)

        self.labelExtractor =  nn.Conv1d(in_channels=128, out_channels=out_features, kernel_size=(1,))

        # self.output_p2 = OuputLayer(in_features,seq_len,out_features)
        # self.output_p5 = OuputLayer(in_features,seq_len,out_features)
        # self.output_p18 = OuputLayer(in_features,seq_len,out_features)

        # for layer in self.modules():
        #     if isinstance(layer, nn.Conv1d):
        #         nn.init.kaiming_normal_(layer.weight.data, nonlinearity='conv1d')
        #         nn.init.zeros_(layer.bias.data)
        #     if isinstance(layer, nn.Linear):
        #         nn.init.kaiming_normal_(layer.weight.data, nonlinearity='linear')
        #         nn.init.zeros_(layer.bias.data)

    def residual_connect(self, x, res):
        return self.relu(self.downsample(x) + res)

    def forward(self, x):
        x = self.bn(x)
        # resid = self.decomp(x)
        # res_0 = x # N, in_feature, 64
        x = self.conv_0(x)
        # res_1 = x # N, 128, 64
        x = self.conv_1(x)
        x = self.dropout(x)
        # res_2 = x # N, 256, 64
        x = self.conv_2(x)

        x = self.seqExtractor1(x)
        # x = self.dropout(x)
        x = self.conv_3(x)
        x = self.dropout(x)

        x = self.conv_4(x)

        x = self.seqExtractor2(x)

        x = self.labelExtractor(x).permute(0, 2, 1)


        # pred_p2 = self.output_p2(x, res_0, res_1, res_2)
        # pred_p5 = self.output_p5(x, res_0, res_1, res_2)
        # pred_p18 = self.output_p18(x, res_0, res_1, res_2)
        # return pred_p2, pred_p5, pred_p18

        return x

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
        tsboard_path = "/home/yby/SGD-HFT-Intern/Projects/T0/CNN/tensorboard_logs/biclsall_matrix_focal"
        if not os.path.exists(tsboard_path):
            os.makedirs(tsboard_path)
        WRITER = SummaryWriter(log_dir=tsboard_path)

    model_path = '/home/yby/SGD-HFT-Intern/Projects/T0/CNN/train_dir_0/model/CNN/param_biclsall_matrix_focal/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if Resume:
        model_name = f'CNNLstmCLS_epoch_7_bs8000_sl64_ts5.pth.tar'
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

    loss_fn = MatrixCrossEntropy(local_rank = local_rank)
    # lossfn_p18 = FocalLoss(alpha = 0.08).cuda()
    # lossfn_p5 = FocalLoss(alpha = 0.04).cuda()
    # lossfn_p2 = FocalLoss(alpha = 0.03).cuda()
    # lossfn_p18 = nn.CrossEntropyLoss().cuda()
    # lossfn_p5 = nn.CrossEntropyLoss().cuda()
    # lossfn_p2 = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(ddp_net.parameters(), lr = 1e-4, weight_decay=1e-3)
    # optimizer = SGD(ddp_net.parameters(), lr = 1e-3, momentum = 0.2, weight_decay=0.01)

    # if prior_epochs != 0:
    #     optimizer.load_state_dict(model_data['optimizer_dict'])
        # scheduler.load_state_dict((model_data['scheduler_dict']))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    ddp_net.train()
    local_rank_id = world_dict[local_rank]
    local_rank_id.sort()
    len_train = int(len(local_rank_id) * 0.7)
    train_ids = local_rank_id[:len_train]
    val_ids = local_rank_id[len_train:]

    train_dataset = HFDatasetBi(local_ids = train_ids,
                              shard_dict = shard_dict,
                              batch_size=BATCH_SIZE,
                              seq_len=SEQ_LEN,
                              time_step = TIMESTEP)

    val_dataset = HFDatasetBi(local_ids=val_ids,
                               shard_dict=shard_dict,
                               batch_size=BATCH_SIZE,
                               seq_len=SEQ_LEN,
                               time_step=TIMESTEP)

    LOGGER.info(f'local rank {local_rank}: Dataset is prepared.')
    LOGGER.info(f'local rank {local_rank}: GPU calculating.')
    datalen = len(train_dataset)
    for epoch_idx in range(prior_epochs, EPOCHS):
        ddp_net.train()

        # total_loss_p2 = 0.0
        # total_loss_p5 = 0.0
        # total_loss_p18 = 0.0
        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_dataset):
            optimizer.zero_grad()
            # h_p2, h_p5, h_p18 = net(x.permute(0,2,1).to(local_rank))
            h = net(x.permute(0,2,1).to(local_rank))

            loss = loss_fn(h, y.to(local_rank))

            total_loss += loss.item()
            # loss_p2 = lossfn_p2(h_p2, y[..., 0].cuda().long())
            # loss_p5 = lossfn_p5(h_p5, y[..., 1].cuda().long())
            # loss_p18 = lossfn_p18(h_p18, y[..., 2].cuda().long())
            # total_loss_p2 += loss_p2.item()
            # total_loss_p5 += loss_p5.item()
            # total_loss_p18 += loss_p18.item()

            # loss = 1/9 * loss_p2 + 3/9 * loss_p5 + 5/9 * loss_p18

            # total_loss += loss.item()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(ddp_net.parameters(), max_norm=1e3)
            optimizer.step()


            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                LOGGER.info(f"Epoch {epoch_idx}: {batch_idx + 1} / {datalen}), loss {loss.item()},")

        scheduler.step()

        if local_rank == 0:
            dist.barrier()
            WRITER.add_scalars(main_tag='LOSS--TRAIN',
                               tag_scalar_dict={
                                   'TOTAL': total_loss / datalen,
                                   # 'p2':total_loss_p2 / datalen,
                                   # 'p5':total_loss_p5 / datalen,
                                   # 'p18':total_loss_p18 / datalen
                               },
                               global_step=epoch_idx + 1)
            for name, param in ddp_net.named_parameters():
                WRITER.add_histogram(name, param.clone().cpu().data.numpy(), epoch_idx + 1)
                WRITER.add_histogram(name + '/grad', param.grad.clone().cpu().data.numpy(), epoch_idx + 1)

            if validation:
                # val_result = validate(net, val_dataset, [lossfn_p2, lossfn_p5, lossfn_p18], local_rank)
                val_result = validate(net, val_dataset, loss_fn, local_rank)
                # total_loss_val, p2_loss_val, p5_loss_val, p18_loss_val = \
                #     val_result['total_loss'], val_result['p2']['loss'], val_result['p5']['loss'], val_result['p18']['loss']
                total_loss_val = val_result['total_loss']
                WRITER.add_scalars(main_tag='LOSS--EVAL',
                                   tag_scalar_dict={
                                       'TOTAL': total_loss_val,
                                       # 'p2': p2_loss_val,
                                       # 'p5': p5_loss_val,
                                       # 'p18': p18_loss_val
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
                                       'recall': p18_recall_val[1],
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
    # total_loss_p5 = 0.0
    # total_loss_p18 = 0.0
    total_loss = 0.0

    summary_p2 = MetricRUC(name = 'p2', classnum=OUTPUT_SIZE//3)
    summary_p5 = MetricRUC(name = 'p5', classnum=OUTPUT_SIZE//3)
    summary_p18 = MetricRUC(name = 'p18', classnum=OUTPUT_SIZE//3)
    total_loss = 0
    with torch.no_grad():
        for x, y in validation_dataset:


            h = net(x.permute(0, 2, 1).to(local_rank))

            h_p2 = h[:, 0, :].squeeze(1)
            h_p5 = h[:, 1, :].squeeze(1)
            h_p18 = h[:, 2, :].squeeze(1)

            loss = lossfn(h, y.to(local_rank))
            total_loss += loss.item()

            y_p2 = y[..., 0]
            y_p5 = y[..., 1]
            y_p18 = y[..., 2]
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

        recall, precision, accuracy, F1 = summary_p2.summary()
        result_dict['total_loss'] = total_loss / len(validation_dataset)
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