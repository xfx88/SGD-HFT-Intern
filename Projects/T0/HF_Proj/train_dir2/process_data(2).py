import datetime

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
from joblib import Parallel, delayed
from scipy.stats import spearmanr
import torchmetrics


col_factors = ['date', 'time', 'timeidx', 'price', 'vwp', 'ask_price', 'bid_price', 'ask_price2', 'bid_price2',
               'ask_price4', 'bid_price4', 'ask_price8', 'bid_price8', 'spread', 'tick_spread',
               'ref_ind_0', 'ref_ind_1', 'ask_weight_14', 'ask_weight_13', 'ask_weight_12', 'ask_weight_11',
               'ask_weight_10', 'ask_weight_9',
               'ask_weight_8', 'ask_weight_7', 'ask_weight_6', 'ask_weight_5', 'ask_weight_4',
               'ask_weight_3', 'ask_weight_2', 'ask_weight_1', 'ask_weight_0', 'bid_weight_0',
               'bid_weight_1', 'bid_weight_2', 'bid_weight_3', 'bid_weight_4', 'bid_weight_5',
               'bid_weight_6', 'bid_weight_7', 'bid_weight_8', 'bid_weight_9', 'bid_weight_10',
               'bid_weight_11', 'bid_weight_12', 'bid_weight_13', 'bid_weight_14', 'ask_dec', 'bid_dec',
               'ask_inc', 'bid_inc', 'ask_inc2', 'bid_inc2', 'preclose', 'limit', 'turnover']

factor_ret_cols = ['timeidx', 'price', 'vwp', 'spread', 'tick_spread', 'ref_ind_0', 'ref_ind_1',
                   'ask_weight_14', 'ask_weight_13', 'ask_weight_12', 'ask_weight_11',
                   'ask_weight_10', 'ask_weight_9', 'ask_weight_8', 'ask_weight_7',
                   'ask_weight_6', 'ask_weight_5', 'ask_weight_4', 'ask_weight_3',
                   'ask_weight_2', 'ask_weight_1', 'ask_weight_0', 'bid_weight_0',
                   'bid_weight_1', 'bid_weight_2', 'bid_weight_3', 'bid_weight_4',
                   'bid_weight_5', 'bid_weight_6', 'bid_weight_7', 'bid_weight_8',
                   'bid_weight_9', 'bid_weight_10', 'bid_weight_11', 'bid_weight_12',
                   'bid_weight_13', 'bid_weight_14', 'ask_dec', 'bid_dec', 'ask_inc',
                   'bid_inc', 'ask_inc2', 'bid_inc2', '10']

path = '/sgd-data/t0_data/echo_date/cnn_data/'

tick_nums = 200
minibatch_size = 150
epochs = 200

factor_numbers = 43
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12357"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,8,9"


def log(x, output=False):
    """
    add time before print
    """
    if output:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), x)

def get_samples(key, tick_nums):
    rs = ut.redis_connection()
    data = ut.read_data_from_redis(rs, key)[factor_ret_cols].fillna(0)
    data = data.values.astype(np.float32)
    if len(data) < tick_nums:
        return
    zero_tensor = torch.zeros((tick_nums - 1, len(factor_ret_cols)))
    data_t = torch.cat((zero_tensor, torch.tensor(data)))
    data2 = data_t.unfold(0, tick_nums, 2).unsqueeze(dim = 3)
    rs.close()
    # log("处理完成 ", key, os.getpid())
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

def classify_tensor(labels):
    for i in range(len(labels)):
        if abs(labels[i].item()) >= 0.0025:
            labels[i] = torch.tensor(2.0)
        elif abs(labels[i].item()) >= 0.0015:
            labels[i] = torch.tensor(1.0)
        else:
            labels[i] = torch.tensor(0)
    return labels


def train(rank, train_redis_keys, criterion, epoch, cnn_model, optimizer):
    # 获取数据
    torch.cuda.set_device(rank)
    # test_acc = torchmetrics.Accuracy(average='none', num_classes=3)
    # test_recall = torchmetrics.Recall(average='none', num_classes=3)
    # test_precision = torchmetrics.Precision(average='none', num_classes=3)
    # test_auc = torchmetrics.AUROC(average="macro",num_classes=3)
    batch_size = 5
    ranks_number = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    rank_batch = list(split(train_redis_keys, ranks_number))
    rs = ut.redis_connection()
    total_loss = 0.0
    for i in range(0, len(rank_batch[rank]), batch_size):
        if i + batch_size > len(rank_batch[rank]):
            batch_keys = rank_batch[rank][i:]
        else:
            batch_keys = rank_batch[rank][i:i + batch_size]
        batch_keys.sort(key=lambda x: str(x).split('_')[-2])
        process = []

        log("rank {}; fetching_data {}".format(rank, batch_keys))
        for key in batch_keys:
            res = get_samples(key, tick_nums)
            process.append(res)
        batch_data = torch.cat(process)
        inputs = batch_data[:, :factor_numbers, :, :]
        log("rank {}; classify_tensor...".format(rank))
        labels = classify_tensor(batch_data[:, factor_numbers, -1, :]).flatten()
        optimizer.zero_grad()
        log("rank {}; gpu_calculating...".format(rank))
        y_pred = cnn_model(inputs).type(torch.DoubleTensor)
        labels = labels.type(torch.LongTensor)
        log("rank {}; labels to gpu...".format(rank))
        labels = labels.to(f"cuda:{rank}").type(torch.LongTensor)
        log("rank {}; criterion...".format(rank))
        loss = criterion(y_pred, labels)
        _, train_pred = torch.max(y_pred, 1)
        total_loss += loss.item() * batch_size

        # test_auc.update(train_pred, labels)
        log("rank {}; backward...".format(rank))
        loss.backward()
        log("rank {}; step...".format(rank))
        optimizer.step()
        # log('rank: {}, epoch: {}, batch: {}-{}, precision: {}'.format(
        #     rank, epoch, i, i+batch_size, loss.item()
        # ))
        # log("rank {}; test_precision, test_recall...".format(rank))
        log(f'rank: {rank}, epoch: {epoch}, batch:{i}-{i+batch_size}, loss: {loss.item()}', True)

    # total_precision = test_precision.compute()
    # total_recall = test_recall.compute()

    # total_auc = test_auc.cuda().compute()
    ut.save_data_to_redis(rs, f"{epoch}_{rank}_classify_train_loss", total_loss/len(rank_batch[rank]))
    # test_auc.reset()
    # test_recall.reset()
    # test_precision.reset()
    model_path = f'model/classify/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if rank == 0:
        torch.save({
            'epoch': epoch,
            'model': cnn_model,
            'optimizer': optimizer,
            'state_dict': cnn_model.state_dict(),
            'optimizer_dict': optimizer.state_dict(),
        }, os.path.join(model_path, f"cnn_model_epoch_{epoch}_ddp.pth.tar"))
    rs.close()
    return cnn_model, optimizer


def model_dev(validated_keys, cnn_model, rank, epoch, criterion):
    cnn_model.eval()
    rs = ut.redis_connection()
    ranks_number = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    rank_batch = list(split(validated_keys, ranks_number))
    batch_size = 5
    total_loss = 0.0
    # test_acc = torchmetrics.Accuracy(average='none', num_classes=3)
    # test_recall = torchmetrics.Recall(average='none', num_classes=3)
    # test_precision = torchmetrics.Precision(average='none', num_classes=3)
    # test_auc = torchmetrics.AUROC(average="macro", num_classes=3)

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
        labels = classify_tensor(batch_data[:, factor_numbers, -1, :]).flatten()
        y_pred = cnn_model(inputs).type(torch.DoubleTensor)
        _, train_pred = torch.max(y_pred, 1)
        labels = labels.to(f"cuda:{rank}").type(torch.LongTensor)

        # test_auc.update(train_pred, labels)
        loss = criterion(y_pred, labels)
        total_loss += loss.detach().cpu().item() * batch_size
        log(f'validation:  rank: {rank}, epoch: {epoch},batch:{i}-{i + batch_size},loss: {loss.item()}', True
            # f',precision : {test_precision(train_pred, labels)},'
            # f'recall:{test_recall(train_pred, labels)}'
        )

    total_loss /= len(rank_batch[rank])
    # total_precision = test_precision.cuda().compute()
    # total_recall = test_recall.cuda().compute()
    # total_auc = test_auc.cuda().compute()
    ut.save_data_to_redis(rs, f"{epoch}_{rank}_classify_valid_loss", total_loss)
    # test_auc.reset()
    # test_recall.reset()
    # test_precision.reset()
    rs.close()
    return total_loss


class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(factor_numbers, 16, kernel_size=(3, 1), padding=(1, 0))
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0))
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0))
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0))
        self.conv6 = torch.nn.Conv2d(128, 4, kernel_size=1)
        self.fc = torch.nn.Linear(4 * 200, 3)

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
        # log("conv1:", x.shape)
        x = F.relu(self.conv2(x))
        # log("conv2:", x.shape)
        x = F.relu(self.conv3(x))
        # log("conv3:", x.shape)
        x = F.relu(self.conv4(x))
        # log("conv4:", x.shape)
        # x = F.relu(self.conv5(x))
        # log("conv5:", x.shape)
        x = self.conv6(x)
        # log("conv6:", x.shape)
        x = x.view(x.shape[0], -1)
        # log("conv6_:", x.shape)
        x = self.fc(x)
        return x


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def model_explain(model_name, train_redis_keys):
    # images, labels = train_set.getbatch(img_indices)
    torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    model_data = torch.load(model_name)
    model = model_data["model"]
    model_state_dict = model_data["state_dict"]
    model.load_state_dict(model_state_dict)
    res = get_samples(train_redis_keys[0], 200)

    inputs = res[199][:factor_numbers, :, :]
    labels = res[199][factor_numbers, -1, :]

    x = inputs
    y = labels
    epoch = 500
    param_sigma_multiplier = 0.01
    model.eval()

    from torch.autograd import Variable
    mean = 0
    sigma_channels = param_sigma_multiplier /((torch.max(x, dim=1)[0] - torch.min(x, dim=1)[0]))
    sigma_x = []
    for sss in sigma_channels:
        if sss.item() != torch.inf:
            sigma_x.append(sss)
            continue
        sigma_x.append(torch.Tensor([0]))
    # sigma = param_sigma_multiplier / (torch.max(x) - torch.min(x)).item()
    smooth = np.zeros(x.cuda().unsqueeze(0).size())
    for i in range(1, epoch+1):
        # call Variable to generate random noise
        # 创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致，且无内容。
        noise_g = []
        for index, sigma in enumerate(sigma_channels):
            if sigma == torch.inf:
                sigma = sigma_x[index]
            noise = Variable(x.data.new(x.size()[1:]).normal_(mean, sigma.item()**2)).unsqueeze(0)
            noise_g.append(noise)
        noise_g = torch.cat(noise_g, dim=0)
        # x_mod = x.cuda()+noise_g.cuda()
        x_mod = (x+noise_g).unsqueeze(0).cuda()
        x_mod.requires_grad_()
        y_pred1 = model(x_mod)
        y_pred2 = model(x.unsqueeze(0))
        log("y_pred1:", y_pred1, "y_pred2:", y_pred2)

        loss_func = torch.nn.MSELoss()
        loss = loss_func(y_pred1, y.cuda().unsqueeze(0))
        loss.backward()
        smooth += x_mod.grad.abs().detach().cpu().data.numpy()
    smooth = normalize(smooth / epoch)
    smooth_sq = torch.tensor(smooth).squeeze(0).squeeze(-1)
    result = smooth_sq.mean(dim = 1)


def plot_loss_cross(epochs=70, ranks_number=6):
    import matplotlib.pyplot as plt
    rs = ut.redis_connection()
    total_train_loss = []
    total_valid_loss = []
    for i in range(1, epochs):
        train_epoch_total_loss = 0
        valid_epoch_total_loss = 0
        for j in range(0, ranks_number):
            train_loss = ut.read_data_from_redis(rs, f"{i}_{j}_classify_train_loss")
            train_epoch_total_loss += train_loss
            valid_loss = ut.read_data_from_redis(rs, f"{i}_{j}_classify_valid_loss")
            valid_epoch_total_loss += valid_loss
        total_train_loss.append(train_epoch_total_loss / ranks_number)
        total_valid_loss.append(valid_epoch_total_loss / ranks_number)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    start_point = 0
    plt.plot([i for i in range(start_point, epochs-1)], total_train_loss[start_point:], "x-", label="train_loss")
    plt.plot([i for i in range(start_point, epochs-1)], total_valid_loss[start_point:], "+-", label="valid_loss")

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()


def model_eval(rank, world_size, test_redis_keys, model_name):
    """
    模型测试
    """
    torch.cuda.set_device(rank)
    log(f"model_eval rank: {rank}", True)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    model_data = torch.load(model_name)
    model = model_data["model"]
    model_state_dict = model_data["state_dict"]
    model.load_state_dict(model_state_dict)
    model.eval()
    batch_size = 5
    y_preds_all = []
    labels_all = []
    for i in range(0, len(test_redis_keys), batch_size):
        if i + batch_size > len(test_redis_keys):
            batch_keys = test_redis_keys[i:]
        else:
            batch_keys = test_redis_keys[i:i + batch_size]
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
        torch.cuda.empty_cache()
        y_preds_all.append(y_pred.detach().numpy())
        labels_all.append(labels.detach().numpy())
    y_preds_all = np.concatenate(y_preds_all)
    labels_all = np.concatenate(labels_all)
    spm = spearmanr(y_preds_all, labels_all)
    log('rank: {}, spm: {}'.format(rank, spm), True)


def run_train(rank, world_size, train_redis_keys, RESUME=None):
    log(f"rank, world_size: {rank}, {world_size}", True)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    # print(local_rank)

    cnn_model = CNNModel().cuda(device=rank)
    cnn_model = torch.nn.parallel.DistributedDataParallel(cnn_model, device_ids=[rank])
    # cirterion = torch.nn.MSELoss(size_average=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-4)
    start_epoch = 0
    if RESUME:
        # 如果是恢复训练，则加载模型
        checkpoint = torch.load(RESUME)  # 加载断点
        cnn_model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_dict'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch

    # 随机取3000个样本作为验证集
    rs = ut.redis_connection()
    if rank == 0:
        import random
        validated_keys = random.sample(train_redis_keys, 3000)
        train_redis_keys = list(set(train_redis_keys) - set(validated_keys))
        ut.save_data_to_redis(rs, "train_redis_keys_classify", train_redis_keys)
        ut.save_data_to_redis(rs, "valid_redis_keys_classify", random.sample(validated_keys, 3000))
    else:
        while not rs.get("train_redis_keys_classify") or not rs.get("valid_redis_keys_classify"):
            log("wait for keys", True)
            time.sleep(0.1)
        train_redis_keys = ut.read_data_from_redis(rs, "train_redis_keys_classify")
        validated_keys = ut.read_data_from_redis(rs, "valid_redis_keys_classify")
    for epoch in range(start_epoch+1, epochs+1):
        a = time.time()
        train(rank, train_redis_keys, criterion, epoch, cnn_model, optimizer)
        model_dev(validated_keys, cnn_model, rank, epoch, criterion)
        log(f"calculate one epoch time: {time.time() - a}", True)
    rs.delete("valid_redis_keys_classify")
    rs.close()


def main():
    world_size = 5
    user = 18123610100
    pw = 123456
    rq.init(user, pw)

    path = '/sgd-data/t0_data/500factors'
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

    # RESUME = "/home/zjr/cnn_test/output/20220217/cnn_model_epoch_99_ddp.pth.tar"
    RESUME = ""

    mp.spawn(run_train,
             args=(world_size, train_redis_keys, RESUME,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()
