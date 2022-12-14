import paramiko
import pandas as pd
import numpy as np
import os
from collections import defaultdict, namedtuple
import math
from datetime import datetime

import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from t0_backtest_view_sdk.t0_view_sdk import t0_viewer
from src.dataset3 import HFDatasetCls
from label_extractor import label_extractor


factor_cols = ['timeidx','price_pct','vwp_pct','price','currentRet','spread','tick_spread','ref_ind_0','ref_ind_1',
               'ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
               'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
               'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
               'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10',
               'ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','turnover','circulation_mv']

tick_cols = ['ask_price1', 'ask_volume1', 'ask_price2', 'ask_volume2',
             'ask_price3', 'ask_volume3', 'ask_price4', 'ask_volume4', 'ask_price5',
             'ask_volume5', 'ask_price6', 'ask_volume6', 'ask_price7', 'ask_volume7',
             'ask_price8', 'ask_volume8', 'ask_price9', 'ask_volume9', 'ask_price10',
             'ask_volume10', 'bid_price1', 'bid_volume1', 'bid_price2',
             'bid_volume2', 'bid_price3', 'bid_volume3', 'bid_price4', 'bid_volume4',
             'bid_price5', 'bid_volume5', 'bid_price6', 'bid_volume6', 'bid_price7',
             'bid_volume7', 'bid_price8', 'bid_volume8', 'bid_price9', 'bid_volume9',
             'bid_price10', 'bid_volume10']

pred_cols = ['subcls_2', 'subcls_5', 'subcls_18', 'subcls_diff']
# factor_ret_cols = ['timeidx','price_pct','vwp_pct','spread','tick_spread','ref_ind_0','ref_ind_1','ask_weight_14',
#                    'ask_weight_13','ask_weight_12','ask_weight_11','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
#                    'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
#                    'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
#                    'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','bid_weight_11','bid_weight_12','bid_weight_13',
#                    'bid_weight_14','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','turnover',
#                    'subcls_2','subcls_5','subcls_18']

"""
eg1 = EgBar('000009', '20211020', '10:45:18') # 7????????????????????????
eg2 = EgBar('000009', '20211020', '10:25:57')  # 7????????????????????????
eg3 = EgBar('000581', '20211026', '09:32:48')  # 19????????????????????????
eg4 = EgBar('603290', '20211015', '09:44:00')  # 390?????????????????????????????????
eg5 = EgBar('603290', '20211015', '09:44:29')  # 390????????????????????????????????????
eg6 = EgBar('603290', '20211015', '09:47:07')  # 390???????????????????????????
"""
EgBar = namedtuple('EgBar', ['stock_id', 'date', 'TIME'])

DATA_PATH = "/home/yby/SGD-HFT-Intern/Projects/T0/Data_labels"

GLOBAL_SEED = 2098

BATCH_SIZE = 10000
RESUME = None

INPUT_SIZE = 79
OUTPUT_SIZE = 20
SEQ_LEN = 64
TIMESTEP = 5

LABEL_IDX = 1

EPOCH_IDX = 19

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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

        # self.bn = nn.BatchNorm1d(in_features)

        _kernel_size = 3

        _layers = []
        # _levels = [INPUT_SIZE, 64, 96, 128, 64, 96, 128]
        _levels = [INPUT_SIZE] * 8
        for i in range(len(_levels)):
            _dilation_size = 1 << max(0, (i - (len(_levels) - 3)))
            _input_size = in_features if i == 0 else _levels[i - 1]
            _output_size = _levels[i]
            _layers.append(CNNBlock(in_channels = _input_size,
                                    out_channels = _output_size,
                                    kernel_size = _kernel_size,
                                    dilation_size = _dilation_size))

        self.network = nn.Sequential(*_layers)

        self.featureExtractor = nn.Conv1d(in_channels=_levels[-1],
                                          out_channels=OUTPUT_SIZE // 4,
                                          kernel_size=(61,))
        # self.sigmoid = nn.Sigmoid()


    def forward(self, x: torch.Tensor):

        # x = self.bn(x)
        x = self.network(x)
        x = self.featureExtractor(x)
        # x = self.sigmoid(x)

        return x.permute(0,2,1)


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


class RemoteSrc:
    REMOTE_PATH = "/sgd-data/data/stock/"
    TEMP = "/home/yby/SGD-HFT-Intern/Projects/T0/CNN/backtest_temp/"

    def __init__(self):
        self._client = paramiko.Transport(("192.168.1.147", 22))
        self._client.connect(username="sgd", password="sgd123")
        self._SFTP = paramiko.SFTPClient.from_transport(self._client)
        if not os.path.exists(self.TEMP):
            os.mkdir(self.TEMP)

        self.dict_stocksPerDay = defaultdict(list)

    def get_raw_bars(self, ticker, date):

        local_path = f"{self.TEMP}{ticker}_{date}.csv.gz"

        if not os.path.exists(local_path):
            files_currentDay = self._SFTP.listdir(f"{self.REMOTE_PATH}{date}/tick_csv/")
            if date in self.dict_stocksPerDay.keys():
                stocks_currentDay = self.dict_stocksPerDay[date]
            else:
                stocks_currentDay = [s[:6] for s in files_currentDay]

            file_idx = stocks_currentDay.index(ticker)

            self._SFTP.get(remotepath=f"{self.REMOTE_PATH}{date}/tick_csv/{files_currentDay[file_idx]}",
                           localpath=local_path)

        data = pd.read_csv(local_path)
        data['server_time'] = pd.to_datetime(data.server_time)
        data['local_time'] = data['server_time']
        data['time'] = data.apply(lambda x: str(x['server_time'].time()), axis = 1)

        return data

class Predict:
    def __init__(self):
        self._load_model()
        self.model.eval()
        self.remote_server = RemoteSrc()

    def _load_model(self):
        model_path = '/home/yby/SGD-HFT-Intern/Projects/T0/CNN/model/CNN_param_cls5all_relu_v2_manu'
        model_name = f'CNNLstmCLS_epoch_25_bs10000_sl64_ts2.pth.tar'
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
        raw_data = self.remote_server.get_raw_bars(ticker = stock_id, date = date).set_index("time")

        np_data = data[factor_cols + tick_cols + pred_cols].values.astype(np.float32)
        np_data = torch.from_numpy(np_data)

        data = data.reset_index()
        data = data.set_index('time')

        sp_dataset = HFDatasetCls(np_data, LEN_SEQ=SEQ_LEN, batch_size = 8000, time_step=1)

        p2_res_list, p2_true = [], []
        p5_res_list, p5_true = [], []
        p18_res_list, p18_true = [], []
        diff_res_list, diff_true = [], []
        for x, y in sp_dataset:
            # p_18_pred = self.model(x.permute(0, 2, 1).cuda())
            p_res = self.model(x.permute(0, 2, 1).cuda())

            p2_res_list.append(p_res[:, 0, :]), p2_true.append(y[:, 0])
            p5_res_list.append(p_res[:, 1, :]), p5_true.append(y[:, 1])
            p18_res_list.append(p_res[:, 2, :]), p18_true.append(y[:, 2])
            diff_res_list.append(p_res[:, 3, :]), diff_true.append(y[:, 3])

        p_2_pred = torch.cat(p2_res_list)
        p_5_pred = torch.cat(p5_res_list)
        p_18_pred = torch.cat(p18_res_list)
        diff_pred = torch.cat(diff_res_list)
        p2_true = torch.cat(p2_true)
        p5_true = torch.cat(p5_true)
        p18_true = torch.cat(p18_true)
        diff_true = torch.cat(diff_true)


        p_2_pred_prob, p_2_pred = label_extractor(p_2_pred, p2_true, cls_num=5)
        p_5_pred_prob, p_5_pred = label_extractor(p_5_pred, p5_true, cls_num=5)
        p_18_pred_prob, p_18_pred = label_extractor(p_18_pred, p18_true, cls_num=5)
        diff_pred_prob, diff_pred = label_extractor(diff_pred, diff_true, cls_num=5)

        pred_y = np.concatenate([p_2_pred, p_5_pred, p_18_pred, diff_pred, p_2_pred_prob, p_5_pred_prob, p_18_pred_prob, diff_pred_prob], axis = 1)
        pred_y = pd.DataFrame(pred_y, index = data.index, columns = ['cls_2_pred', 'cls_5_pred','cls_18_pred','cls_diff_pred',
                                                                     'cls_2_prob', 'cls_5_prob','cls_18_prob','cls_diff_prob'])
        # pred_y = pd.DataFrame(pred_y, index = data.index, columns = ['cls_5_pred'])
        pred_y = pd.merge(pred_y, data[['subcls_2','subcls_5','subcls_18','subcls_diff', 'p_2', 'p_5', 'p_18', 'p_diff']], left_index = True, right_index = True)
        res = pred_y.merge(raw_data, how = "inner", left_index=True, right_index=True)
        res["currentRet"] = res["last"].pct_change()

        res.reset_index(inplace = True)

        res.loc[:, 'predictions'] = list(res[pred_cols].values)

        times = 2 if stock_id.startswith('688') else 1
        res['Direction'] = res['predictions'].apply(lambda x: "Buy" if (sum(map(lambda k: int(k) == 4, x)) == 2) \
            else \
            ( "Sell" if (sum(map(lambda k: int(k) == 2, x)) == 2) else None))
        res['pos'] = res['predictions'].apply(lambda x: gen_open_pos(x, times=times))
        res["prevPos"] = res.pos.shift(1)

        res["pos"] = res.apply(lambda x: 0 if ((x["pos"] > 0 and (not x["prevPos"] > 0) and x["currentRet"] > 0.01)
                                               or (x["pos"] < 0 and (not x["prevPos"] > 0) and x["currentRet"] < -0.01)) else x["pos"], axis = 1)
        res.drop("prevPos", axis = 1, inplace = True)
        res.loc[res["pos"] == 0, "Direction"] = 0
        res['fillPos'] = res.pos.fillna(method='ffill')
        res.loc[~res["pos"].isna(), "cost"] = res.loc[~res["pos"].isna(), "last"]
        res["cost"] = res["cost"].fillna(method='ffill')

        buy_close_condition = (res.fillPos > 0) & ((
                                                       res['predictions'].apply(
                                                           lambda x: sum(
                                                               [i == 1 or i == 2 for i in x]) == 2)) | res.apply(
            lambda x: abs(x["last"] / x["cost"] - 1) >= 0.01, axis=1)
                                                   )
        sell_close_condition = (res.fillPos < 0) & ((
                                                        res['predictions'].apply(
                                                            lambda x: sum(
                                                                [i == 3 or i == 4 for i in x]) == 2)) | res.apply(
            lambda x: abs(x["last"] / x["cost"] - 1) >= 0.01, axis=1)
                                                    )
        res.loc[(buy_close_condition | sell_close_condition), 'pos'] = 0
        res.pos = res.pos.fillna(method='ffill')

        res.loc[:, 'comm_signal'] = 0.

        res.loc[res['pos'] > 0, 'comm_price'] = res.loc[res['pos'] > 0, 'ask_price1']
        res.loc[sell_close_condition, 'comm_price'] = res.loc[sell_close_condition, 'last']
        res.loc[res['pos'] < 0, 'comm_price'] = res.loc[res['pos'] < 0, 'bid_price1']
        res.loc[buy_close_condition, 'comm_price'] = res.loc[buy_close_condition, 'last']

        # res.loc[(res['pos'] > 0 | sell_close_condition), 'comm_price'] = res.loc[
        #     (res['pos'] > 0 | sell_close_condition), 'ask_price3']
        # res.loc[(res['pos'] < 0 | buy_close_condition), 'comm_price'] = res.loc[
        #     (res['pos'] < 0 | buy_close_condition), 'bid_price3']

        res.drop(['predictions'], axis=1, inplace=True)

        res['custom_cls2 | cls2 pred'] = res.apply(lambda x: str(round(x["p_2"], 6)) + " | " + str(x["subcls_2"]) + " | " + str(x["cls_2_pred"]), axis = 1)
        res['custom_cls5 | cls5 pred'] = res.apply(lambda x: str(round(x["p_5"], 6))  + " | " + str(x["subcls_5"]) + " | " + str(x["cls_5_pred"]), axis = 1)
        res['custom_cls18 | cls18 pred'] = res.apply(lambda x: str(round(x["p_18"], 6)) + " | " + str(x["subcls_18"]) + " | " + str(x["cls_18_pred"]), axis = 1)
        res['custom_clsdiff | clsdiff pred'] = res.apply(lambda x: str(round(x["p_diff"], 6)) + " | " + str(x["subcls_diff"]) + " | " + str(x["cls_diff_pred"]), axis = 1)

        res.rename(columns={"cls_2_prob": "custom_cls2 Prob",
                            "cls_5_prob": "custom_cls5 Prob",
                            "cls_18_prob": "custom_cls18 Prob",
                            "cls_diff_prob": "custom_diff Prob",
                            "pos": "custom_POS"
                            }, inplace=True)

        return res



def query(stock_id, date):
    SAVING_PATH = '../Backtest/'
    if not os.path.exists(SAVING_PATH):
        os.makedirs(SAVING_PATH)
    predict = Predict()

    stock_bars:pd.DataFrame = predict.specific_stock(stock_id=stock_id, date=date)
    stock_bars['date'] = stock_bars.apply(lambda x: str(x['server_time'].date()), axis = 1)
    stock_bars.to_csv(f"./tmp/{stock_id}.SSE_{date}.csv", encoding = 'utf-8')

    t0_viewer.upload(model_name=f"wzhh_CLS_{80}", source_files="./tmp/")
    # stock_signals = stock_bars[["date", "time", "code", "Direction"]]
    # stock_bars.drop(["date", "time", "code", "Direction"], axis=1, inplace=True)
    #
    # stock_bars.to_csv("%s%s_%s_EPOCH50.csv" % (SAVING_PATH, stock_id, date), encoding = 'utf-8')
    # stock_signals.to_csv("%s%s_%s_signals_EPOCH50.csv" % (SAVING_PATH, stock_id, date), encoding = 'utf-8')



if __name__ == "__main__":
    query("603290", "20211104")