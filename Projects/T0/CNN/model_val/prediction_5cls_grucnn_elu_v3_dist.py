import sys
sys.path.append("/home/yby/SGD-HFT-Intern/Projects/T0/CNN")

import os
import gc
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import namedtuple
import numpy as np
import pandas as pd
import math
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from src.dataset3 import HFDatasetCls
import utilities as ut
from Backtest import RemoteSrc
from label_extractor import label_extractor

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

RET_COLS = ['cls_5']
pred_cols = ['cls_2_pred', 'cls_5_pred', 'cls_18_pred']

factor_ret_cols = ['timeidx','price_pct','vwp_pct','ref_ind_0','ref_ind_1','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
                   'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
                   'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
                   'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','turnover',
                   'subcls_2','subcls_5','subcls_18']
# factor_ret_cols = ['timeidx','price_pct','vwp_pct','spread','tick_spread','ref_ind_0','ref_ind_1','ask_weight_14',
#                    'ask_weight_13','ask_weight_12','ask_weight_11','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
#                    'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
#                    'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
#                    'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','bid_weight_11','bid_weight_12','bid_weight_13',
#                    'bid_weight_14','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','turnover',
res_col = ['cls_2','cls_5','cls_18',
           'cls_2_pred','cls_5_pred','cls_18_pred']

EgBar = namedtuple('EgBar', ['stock_id', 'date', 'TIME'])

GLOBAL_SEED = 2098

BATCH_SIZE = 5000
RESUME = None

INPUT_SIZE = 34
OUTPUT_SIZE = 15
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
                              kernel_size=(3,),
                              padding=((dilation_size * 2) >> 1,),
                              dilation=(dilation_size,)))

        self.sub_block = nn.Sequential(self.cnn1, nn.ReLU(), self.cnn2, nn.ReLU())
        self.output_relu = nn.ReLU()
        self.resample = weight_norm(nn.Conv1d(in_channels,
                                  out_channels,
                                  kernel_size=(kernel_size,),
                                  padding=(_padding_size,),
                                  dilation=(dilation_size,))) \
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

        # self.bn = nn.BatchNorm1d(in_features)

        _kernel_size = 1

        _layers = []
        _levels = [INPUT_SIZE, 64, 96, 128, 128, 96, 48, 32]
        # _levels = [INPUT_SIZE] * 7
        for i in range(len(_levels)):
            _dilation_size = 1 << max(0, (i - (len(_levels) - 3)))
            _input_size = in_features if i == 0 else _levels[i - 1]
            _output_size = _levels[i]
            _layers.append(CNNBlock(in_channels = _input_size,
                                    out_channels = _output_size,
                                    kernel_size = _kernel_size,
                                    dilation_size = _dilation_size))
            _layers.append(nn.Dropout(0.1))

        self.network = nn.Sequential(*_layers)

        self.featureExtractor = nn.Linear(_levels[-1] * seq_len, OUTPUT_SIZE)
        # self.sigmoid = nn.Sigmoid()


    def forward(self, x: torch.Tensor):

        # x = self.bn(x)
        x = self.network(x)
        x = self.featureExtractor(x.flatten(1))
        x = x.view((-1, 3, 5))
        # x = self.sigmoid(x)

        return x

def load_data(date_key):

    rs = ut.redis_connection(db = 0)
    data = ut.read_data_from_redis(rs, date_key)
    rs.close()
    return data

def load_df_data(eg: EgBar, seq_len = 50):
    df = pd.read_pickle(f"{DATA_PATH}/{eg.stock_id}/{eg.date}.pkl")
    df = df.loc[: eg.TIME]
    partition = df[factor_ret_cols[1:-4]]
    df = df[RET_COLS]
    len_partition = len(partition)
    partition = torch.Tensor(partition.values[-seq_len:]).unsqueeze(0)
    if len(partition[0]) < seq_len:
        partition = F.pad(partition.transpose(1, 2), (seq_len - len_partition, 0), 'constant').transpose(1,2)

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
LABEL_IDX = 1

def gen_open_pos(x, times = 1):

    # if (int(x[LABEL_IDX]) == 4):
    #     return 100 * times
    # elif (int(x[LABEL_IDX]) == 2):
    #     return -100 * times
    if (sum(map(lambda k: int(k) == 4, x)) == 2):
        return 100 * times
    elif (sum(map(lambda k: int(k) == 2, x)) == 2):
        return -100 * times

    return np.nan

class Predict:
    def __init__(self):
        self._load_model()
        self.model.eval()
        self.remote_server = RemoteSrc()

    def _load_model(self):
        model_path = '/home/yby/SGD-HFT-Intern/Projects/T0/CNN/train_dir_0/model/CNN_param_cls5all_matrix_relu_v3_manu'
        model_name = 'CNNLstmCLS_epoch_99_bs10000_sl64_ts2.pth.tar'
        model_data = torch.load(os.path.join(model_path, model_name))
        from collections import OrderedDict
        local_state_dict = OrderedDict()

        for k, v in model_data['state_dict'].items():
            name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
            local_state_dict[name] = v

        model = ConvLstmNet(in_features=INPUT_SIZE,
                            seq_len=SEQ_LEN,
                            out_features=OUTPUT_SIZE // 3)
        model.load_state_dict(local_state_dict)

        self.model = model.cuda()

        self.model.eval()


    def specific_stock(self, stock_id, date):

        data = pd.read_pickle(f"{DATA_PATH}/{stock_id}/{date}.pkl").fillna('0')
        raw_data = self.remote_server.get_raw_bars(ticker=stock_id, date=date).set_index("time")

        np_data = data[factor_ret_cols].values.astype(np.float32)
        np_data = torch.from_numpy(np_data)

        data = data.reset_index()
        data = data.set_index('time')

        sp_dataset = HFDatasetCls(np_data, LEN_SEQ=SEQ_LEN, batch_size = 40000, time_step=1)

        p2_res_list = []
        p5_res_list = []
        p18_res_list = []
        for x, y in sp_dataset:
            # p_18_pred = self.model(x.permute(0, 2, 1).cuda())
            p_res = self.model(x.permute(0, 2, 1).cuda())
            p_2_pred, p_5_pred, p_18_pred = p_res.split(1, dim = 1)

            p2_res_list.append(p_2_pred.squeeze(1))
            p5_res_list.append(p_5_pred.squeeze(1))
            p18_res_list.append(p_18_pred.squeeze(1))

            del x, y
            gc.collect()

        p_2_pred = torch.cat(p2_res_list)
        p_5_pred = torch.cat(p5_res_list)
        p_18_pred = torch.cat(p18_res_list)

        p_2_pred_prob, p_2_pred = label_extractor(p_2_pred, prob_up=[0.6, 0.65], prob_down=[0.6, 0.65], cls_num=5)
        p_5_pred_prob, p_5_pred = label_extractor(p_5_pred, prob_up=[0.6, 0.65], prob_down=[0.6, 0.65], cls_num=5)
        p_18_pred_prob, p_18_pred = label_extractor(p_18_pred, prob_up=[0.6, 0.65], prob_down=[0.6, 0.65], cls_num=5)

        pred_y = np.concatenate([p_2_pred, p_5_pred, p_18_pred, p_2_pred_prob, p_5_pred_prob, p_18_pred_prob], axis=1)
        pred_y = pd.DataFrame(pred_y, index=data.index, columns=['cls_2_pred', 'cls_5_pred', 'cls_18_pred',
                                                                 'cls_2_prob', 'cls_5_prob', 'cls_18_prob'])

        pred_y = pd.merge(pred_y, data[['subcls_2', 'subcls_5', 'subcls_18', 'p_2', 'p_5', 'p_18']], left_index=True,
                          right_index=True)
        res = pred_y.merge(raw_data, how="inner", left_index=True, right_index=True)
        res["currentRet"] = res["last"].pct_change()

        res.reset_index(inplace=True)

        res.loc[:, 'predictions'] = list(res[pred_cols].values)

        times = 2 if stock_id.startswith('688') else 1
        res['pos'] = res['predictions'].apply(lambda x: gen_open_pos(x, times=times))

        res["prevPos"] = res.pos.shift(1)

        try:
            res["pos"] = res.apply(lambda x: 0 if ((x["pos"] > 0 and (not x["prevPos"] > 0) and x["currentRet"] > 0.01)
                                                   or (x["pos"] < 0 and (not x["prevPos"] > 0) and x[
                        "currentRet"] < -0.01)) else x["pos"], axis=1)
        except:
            print("here")

        res.drop("prevPos", axis=1, inplace=True)
        res['fillPos'] = res.pos.fillna(method='ffill')
        res.loc[~res["pos"].isna(), "cost"] = res.loc[~res["pos"].isna(), "last"]
        res["cost"] = res["cost"].fillna(method='ffill')

        buy_close_condition = (res.fillPos > 0) & ((
                                                       res['predictions'].apply(
                                                           lambda x: sum([i == 1 or i == 2 for i in x]) == 2)) | res.apply(
            lambda x: abs(x["last"] / x["cost"] - 1) >= 0.01, axis=1)
                                                   )
        sell_close_condition = (res.fillPos < 0) & ((
                                                        res['predictions'].apply(
                                                            lambda x: sum([i == 3 or i == 4 for i in x]) == 2)) | res.apply(
            lambda x: abs(x["last"] / x["cost"] - 1) >= 0.01, axis=1)
                                                    )

        res.loc[(buy_close_condition | sell_close_condition), 'pos'] = 0
        res.pos = res.pos.fillna(method='ffill')


        res.loc[:, 'comm_signal'] = 0.

        res.loc[res['pos'] > 0, 'comm_price'] = res.loc[res['pos'] > 0, 'ask_price1']
        res.loc[res.pos == 0, 'comm_price'] = res.loc[res.pos == 0, 'last']
        res.loc[res['pos'] < 0, 'comm_price'] = res.loc[res['pos'] < 0, 'bid_price1']
        res.loc[res.pos == 0, 'comm_price'] = res.loc[res.pos == 0, 'last']

        res.drop(['predictions'], axis=1, inplace=True)

        # res['custom_cls2 | cls2 pred'] = res.apply(
        #     lambda x: str(round(x["p_2"], 6)) + " | " + str(x["cls_2"]) + " | " + str(x["cls_2_pred"]), axis=1)
        # res['custom_cls5 | cls5 pred'] = res.apply(
        #     lambda x: str(round(x["p_5"], 6)) + " | " + str(x["cls_5"]) + " | " + str(x["cls_5_pred"]), axis=1)
        # res['custom_cls18 | cls18 pred'] = res.apply(
        #     lambda x: str(round(x["p_18"], 6)) + " | " + str(x["cls_18"]) + " | " + str(x["cls_18_pred"]), axis=1)
        #
        # res.rename(columns={"cls_2_prob": "custom_cls2 Prob",
        #                     "cls_5_prob": "custom_cls5 Prob",
        #                     "cls_18_prob": "custom_cls18 Prob"
        #                     }, inplace=True)

        return res


cols_backtest = ["time", "date", "code", "pos", "comm_signal", "comm_price"]

def specific_stock_batch(stock_ids, start_date, end_date):
    SAVING_PATH = '../prediction/grucnn_elu_v3/5cls_elu_ce_epoch32_p5bound0.6/'
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
            stock_signals.append(predict.specific_stock(stock_id=stock_id, date=date)[cols_backtest])
        stock_signals:pd.DataFrame = pd.concat(stock_signals)

        stock_signals.to_csv("%s%s.csv" % (SAVING_PATH, stock_id), encoding = 'utf-8')

if __name__ == "__main__":
    stocks = ['603290', '603893', '603260', '688029', '600563',
              '603444', '688099', '600556', '603345', '603605',
              '603806', '603486']
    specific_stock_batch(stocks, start_date=20211101, end_date=20211130)
