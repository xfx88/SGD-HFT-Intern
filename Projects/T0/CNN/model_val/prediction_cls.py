import sys
sys.path.append("/home/yby/SGD-HFT-Intern/Projects/T0/CNN")

from datetime import datetime
from collections import namedtuple
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.device('cuda:1')

from src.dataset3 import HFDatasetCls
import utilities as ut
from train_dir_0.fast_soft_sort.pytorch_ops import soft_rank

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

RET_COLS = ['cls_18']
pred_cols = ['cls_18_pred']

factor_ret_cols = ['timeidx','price_pct','vwp_pct','spread','tick_spread','ref_ind_0','ref_ind_1','ask_weight_14',
                   'ask_weight_13','ask_weight_12','ask_weight_11','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
                   'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
                   'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
                   'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','bid_weight_11','bid_weight_12','bid_weight_13',
                   'bid_weight_14','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','turnover',
                   'cls_2','cls_5','cls_18']

# factor_ret_cols = ['timeidx','price','vwp','ask_price','bid_price','ask_price2','bid_price2','ask_price4',
#                    'bid_price4','ask_price8','bid_price8','spread','tick_spread','ref_ind_0','ref_ind_1','ask_weight_14',
#                    'ask_weight_13','ask_weight_12','ask_weight_11','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
#                    'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
#                    'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
#                    'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','bid_weight_11','bid_weight_12','bid_weight_13',
#                    'bid_weight_14','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','preclose','limit','turnover',
#                    'circulation_mv', 'p_2','p_5','p_18','p_diff']
res_col = ['cls_2','cls_5','cls_18',
           'cls_18_pred']

EgBar = namedtuple('EgBar', ['stock_id', 'date', 'TIME'])

GLOBAL_SEED = 2098

BATCH_SIZE = 5000
RESUME = None

INPUT_SIZE = 44
OUTPUT_SIZE = 9
SEQ_LEN = 64
TIMESTEP = 5

# BATCH_SIZE = 10000
# WARMUP = 4000
# RESUME = None
#
# INPUT_SIZE = 43
# OUTPUT_SIZE = 4
# SEQ_LEN = 64
# TIMESTEP = 1

DATA_PATH = "/home/yby/SGD-HFT-Intern/Projects/T0/Data_labels"

class CNNBlock(nn.Module):
    def __init__(self, in_channels: int,
                       out_channels: int,
                       kernel_size: tuple,
                       padding: tuple,
                       dilation: tuple = (1, ),
                       max_pool_kernel: bool or tuple = (2, )):

        super(CNNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              dilation=dilation,
                              kernel_size=kernel_size,
                              padding = padding)
        self.relu = nn.ReLU()
        if max_pool_kernel:
            self.max_pool = nn.MaxPool1d(kernel_size=max_pool_kernel)
        else:
            self.max_pool = None

    def forward(self, x):

        x = self.conv(x)
        x = self.relu(x)

        if self.max_pool:
            x = self.max_pool(x)

        return x


class ConvLstmNet(nn.Module):
    def __init__(self, in_features, seq_len, out_features):
        super(ConvLstmNet, self).__init__()
        # set size
        self.in_features = in_features
        self.seq_len     = seq_len
        self.out_features = out_features

        # self.BN = nn.BatchNorm1d(num_features=in_features)

        self.conv_0 =  CNNBlock(in_channels= in_features,out_channels = 64,kernel_size=(3, ),dilation=(4,),padding=(4, )) # seq: 64 -> 32

        self.conv_1 =  CNNBlock(in_channels= 64,out_channels = 128,kernel_size=(3, ),padding=(1, ), max_pool_kernel=False) # seq: 32 -> 32

        self.conv_2 = CNNBlock(in_channels=128,out_channels=256,kernel_size=(3,),padding=(1,)) # seq: 32 -> 16

        self.conv_3 = CNNBlock(in_channels=256,out_channels=512,kernel_size=(3,),padding=(1,)) # seq: 16 -> 8

        # self.conv_3 = CNNBlock(in_channels=128,out_channels=256,kernel_size=(3,),padding=(1,)) # seq: 32 -> 16

        # self.conv_4 = CNNBlock(in_channels=256,out_channels=512,kernel_size=(1,),padding=(0,)) # seq: 16 -> 8



        # self.downsample1 = nn.Linear(512 * 8, 3)
        # self.downsample2 = nn.Linear(512 * 8, 3)
        self.downsample3 = nn.Linear(512 * 8, 3)

        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight.data, nonlinearity='conv1d')
                nn.init.zeros_(layer.bias.data)
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight.data, nonlinearity='linear')
                nn.init.zeros_(layer.bias.data)

    def forward(self, x):
        # x = self.BN(x)
        conv_feat = self.conv_0(x)
        conv_feat = self.conv_1(conv_feat)
        conv_feat = self.conv_2(conv_feat)
        conv_feat = self.conv_3(conv_feat)
        # conv_feat = self.conv_4(conv_feat)

        conv_feat = conv_feat.squeeze(-1)

        # y_p2 = self.downsample1(conv_feat.flatten(1))
        # y_p5 = self.downsample2(conv_feat.flatten(1))
        y_p18 = self.downsample3(conv_feat.flatten(1))

        return y_p18

def load_data(date_key):

    rs = ut.redis_connection(db = 0)
    data = ut.read_data_from_redis(rs, date_key)
    rs.close()
    return data

def load_df_data(eg: EgBar, seq_len = 50):
    df = pd.read_pickle(f"{DATA_PATH}/{eg.stock_id}/{eg.date}.pkl")
    # query = bytes(query, encoding = "utf-8")
    # rs = ut.redis_connection(db = 0)
    # df = ut.read_data_from_redis(rs, query)
    # rs.close()
    df = df.loc[: eg.TIME]
    partition = df[factor_ret_cols[1:-4]]
    df = df[RET_COLS]
    len_partition = len(partition)
    partition = torch.Tensor(partition.values[-seq_len:]).unsqueeze(0)
    if len(partition[0]) < seq_len:
        partition = F.pad(partition.transpose(1, 2), (seq_len - len_partition, 0), 'constant').transpose(1,2)
    # partition = partition[:, -seq_len :, ...]

    return partition, len_partition, df

def generate_keys(start_date, end_date, stock_id, prefix = 'clslabels'):
    rs = ut.redis_connection(db=0)
    all_redis_keys = rs.keys()
    rs.close()
    if prefix == 'clslabels':
        keys_of_dates = [x for x in all_redis_keys if (len(str(x).split('_')) == 3)
                         and (str(x.decode()).split('_')[0] == prefix)
                         and (str(x.decode()).split('_')[2] <= str(end_date)[4:6])
                         and (str(x.decode()).split('_')[2] >= str(start_date)[4:6])]
        return keys_of_dates

    elif prefix == 'df':
        keys_of_dates = [x for x in all_redis_keys if (len(str(x).split('_')) == 3)
                         and (str(x.decode()).split('_')[0] == prefix)
                         and (str(x.decode()).split('_')[1] == str(end_date))
                         and (str(x.decode()).split('_')[2] == stock_id)]
        return keys_of_dates[0]


def gen_open_pos(x, times = 1):
    # if x[1] == 2 or x[2] == 2:
    if x == 2:
        return 100 * times
    elif x == 1:
        return -100 * times

class Predict:
    def __init__(self):
        self._load_model()
        self.model.eval()

    def _load_model(self):
        model_path = '/home/yby/SGD-HFT-Intern/Projects/T0/CNN/train_dir_0/model/CNN/param_cls18_v2_adam_datamodified/'
        model_name = 'CNNLstmCLS_epoch_39_bs5000_sl64_ts5.pth.tar'
        model_data = torch.load(os.path.join(model_path, model_name))
        from collections import OrderedDict
        local_state_dict = OrderedDict()

        for k, v in model_data['state_dict'].items():
            name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
            local_state_dict[name] = v

        model = ConvLstmNet(in_features=INPUT_SIZE,
                            seq_len=SEQ_LEN,
                            out_features=OUTPUT_SIZE)
        model.load_state_dict(local_state_dict)

        self.model = model.cuda()

        self.model.eval()

    def all_bars(self, start_date, end_date):

        prediction_result_path = 'CNNprediction/param_1/'
        
        all_keys = generate_keys(start_date, end_date, stock_id=None, prefix='numpy')
        sp2 = []
        sp5 = []
        sp18 = []
        sp = [sp2, sp5, sp18]

        for ticker_keys in all_keys:
            y_pred_all = []
            y_all = []

            ticker_key = ticker_keys.decode().split("_")
            ticker = ticker_key[1]
            print(f"Ticker {ticker}: prediction start...")
            month_str = ticker_key[2]
            date_str = f'2021{month_str}'
            result_path = prediction_result_path + date_str + "/"
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            test_data = load_data(ticker_keys)
            test_dataset = HFDatasetCls(test_data, LEN_SEQ=SEQ_LEN, batch_size=5000)

            with torch.no_grad():
                for idx, (x, y) in enumerate(test_dataset):
                    y_pred = self.model(x.permute(0, 2, 1).cuda()).to('cpu')
                    y_pred_all.append(y_pred.detach().numpy())
                    y_all.append(y.detach().numpy())
            y_pred_concat = np.concatenate(y_pred_all)
            y_concat = np.concatenate(y_all)
            for i in range(3):
                sp[i].append(spearmanr(y_pred_concat[:, i], y_concat[:, i])[0])

        for i in range(3):
            print(np.mean(sp[i]))

    def specific_stock(self, stock_id, date):

        data = pd.read_pickle(f"{DATA_PATH}/{stock_id}/{date}.pkl").fillna('0')

        np_data = data[factor_ret_cols].values.astype(np.float32)
        np_data = torch.from_numpy(np_data)

        sp_dataset = HFDatasetCls(np_data, LEN_SEQ=SEQ_LEN, batch_size = 800, time_step=1)

        res_list = []
        for x, y in sp_dataset:
            p_18_pred = self.model(x.permute(0, 2, 1).cuda())
            # p_2_pred, p_5_pred, p_18_pred = self.model(x.permute(0, 2, 1).cuda())
            # p_2_pred = torch.argmax(p_2_pred, dim = 1).unsqueeze(-1).to('cpu').detach().numpy()
            # p_5_pred = torch.argmax(p_5_pred, dim = 1).unsqueeze(-1).to('cpu').detach().numpy()
            p_18_pred = torch.argmax(p_18_pred, dim = 1).unsqueeze(-1).to('cpu').detach().numpy()
            res_list.append(p_18_pred)
        pred_y = np.concatenate(res_list)
        # pred_y = pd.DataFrame(pred_y, index = data.index, columns = ['cls_2_pred', 'cls_5_pred','cls_18_pred'])
        pred_y = pd.DataFrame(pred_y, index = data.index, columns = ['cls_18_pred'])
        result = pd.concat([data, pred_y], axis=1)

        result.reset_index(inplace = True)
        res = result[['time', 'date', 'code', 'price', 'cls_2', 'cls_5', 'cls_18', 'cls_18_pred']]
        # res = result[['time', 'date', 'code', 'price', 'cls_2', 'cls_5', 'cls_18', 'cls_2_pred', 'cls_5_pred','cls_18_pred']]

        res.loc[:, 'predictions'] = list(res[pred_cols].values)

        times = 2 if stock_id.startswith('688') else 1
        res['pos'] = res['predictions'].apply(lambda x: gen_open_pos(x, times = times))

        res['fillPos'] = res.pos.fillna(method='ffill')

        buy_close_condition = (res.fillPos > 0) & (res['predictions'].apply(lambda x: all(x == 0)))
        sell_close_condition = (res.fillPos < 0) & (res['predictions'].apply(lambda x: all(x == 0)))
        res.loc[(buy_close_condition | sell_close_condition), 'pos'] = 0
        res.pos = res.pos.fillna(method='ffill')
        res.loc[res['pos'] > 0, 'comm_signal'] = 0.001
        res.loc[res['pos'] < 0, 'comm_signal'] = 0.001
        res.loc[res['pos'] == 0, 'comm_signal'] = 0.002

        res['comm_price'] = round(res.price * (1 + res['comm_signal']), 2)
        res['comm_price2'] = res.price * (1 + res['comm_signal'])

        res.drop(['cls_2', 'cls_5', 'cls_18', 'cls_18_pred', 'predictions'], axis = 1, inplace = True)

        return res

        # result.reset_index(inplace = True)
        # result["local_time"] = result["date"].str.cat(result["time"], sep=" ")
        # result["server_time"] = result["date"].str.cat(result["time"], sep=" ")
        # result["last"] = result["price"]
        # result.to_csv("sample_quota.csv")

    def specific_bar(self, egs: EgBar or List, seq_len = 20):

        if not isinstance(egs, list):
            egs = [egs]

        res = []
        INPUT_SHAPE = (-1, INPUT_SIZE, SEQ_LEN, 1)
        for eg in egs:
            test_x, len_x, y_true = load_df_data(eg, SEQ_LEN)
            if not len(test_x.size()) == 3:
                raise AttributeError(f"Input must have shape of (batch_size, seq_len, features).\n")

            pred_y = self.model(test_x.permute(0,2,1).cuda()).to('cpu')
            pred_y = pred_y.squeeze(0).detach().numpy()
            print(f"{eg.stock_id}, {eg.date} {eg.TIME} : ")
            print("prediction:", pred_y)
            print("true value:", y_true.values[-1], "\n")



def specific_stock_batch(stock_ids, start_date, end_date):
    SAVING_PATH = '../prediction/v2/CNN_param_adam_p18_resampled/'
    if not os.path.exists(SAVING_PATH):
        os.makedirs(SAVING_PATH)
    predict = Predict()

    for stock_id in stock_ids:
        print(f'------------{stock_id}------------')
        file_names = os.listdir(f"{DATA_PATH}/{stock_id}/")
        dates = [f[:8] for f in file_names if int(f[:8]) <= end_date and int(f[:8]) >= start_date]
        stock_signals = []
        dates.sort()
        for date in dates:
            stock_signals.append(predict.specific_stock(stock_id=stock_id, date=date))
        stock_signals:pd.DataFrame = pd.concat(stock_signals)
        stock_signals.to_csv("%s%s.csv" % (SAVING_PATH, stock_id), encoding = 'utf-8')
        print("\n")

if __name__ == "__main__":
    stocks = ['603290', '603893', '603260', '688029', '600563',
              '603444', '688099', '600556', '603345', '603605',
              '603806', '603486']
    specific_stock_batch(stocks, start_date=20211101, end_date=20211130)

    # predict = Predict()
    # predict.all_bars(start_date=20211001, end_date=20211031)
    # eg1 = EgBar('000009', '20211020', '10:45:18') # 7????????????????????????
    # eg2 = EgBar('000009', '20211020', '10:25:57')  # 7????????????????????????
    # eg3 = EgBar('000581', '20211026', '09:32:48')  # 19????????????????????????
    # eg4 = EgBar('603290', '20211015', '09:44:00')  # 390?????????????????????????????????
    # eg5 = EgBar('603290', '20211015', '09:44:29')  # 390????????????????????????????????????
    # eg6 = EgBar('603290', '20211015', '09:47:07')  # 390???????????????????????????
    # egs = [eg1, eg2, eg3, eg4, eg5, eg6]

    # print("6 level CNN + 1 LSTM RESULT : ")
    # predict.specific_bar(egs)
    # predict.all_bars(start_date=20211001, end_date=20211031)
    # predict.specific_stock(stock_id='603290', date='20211015')