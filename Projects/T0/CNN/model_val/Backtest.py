import paramiko
import pandas as pd
import numpy as np
import os
from collections import defaultdict, namedtuple
from src.dataset3 import HFDatasetCls
from torch.nn.utils import weight_norm
from label_extractor import *
import math
from datetime import datetime

import torch
import torch.nn.functional as F

factor_ret_cols = ['timeidx','price_pct','vwp_pct','spread','tick_spread','ref_ind_0','ref_ind_1','ask_weight_14',
                   'ask_weight_13','ask_weight_12','ask_weight_11','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
                   'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
                   'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
                   'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','bid_weight_11','bid_weight_12','bid_weight_13',
                   'bid_weight_14','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','turnover',
                   'cls_2','cls_5','cls_18']

"""
eg1 = EgBar('000009', '20211020', '10:45:18') # 7元，盘口放量突破
eg2 = EgBar('000009', '20211020', '10:25:57')  # 7元，盘口放量突破
eg3 = EgBar('000581', '20211026', '09:32:48')  # 19元，盘口放量下跌
eg4 = EgBar('603290', '20211015', '09:44:00')  # 390元，回调结束，上涨开始
eg5 = EgBar('603290', '20211015', '09:44:29')  # 390元，盘口小放量，向上突破
eg6 = EgBar('603290', '20211015', '09:47:07')  # 390元，大涨结束，回调
"""
pred_cols = ['cls_2_pred', 'cls_5_pred', 'cls_18_pred']
EgBar = namedtuple('EgBar', ['stock_id', 'date', 'TIME'])

DATA_PATH = "/home/yby/SGD-HFT-Intern/Projects/T0/Data_labels"

GLOBAL_SEED = 2098

BATCH_SIZE = 5000
RESUME = None

INPUT_SIZE = 44
OUTPUT_SIZE = 9
SEQ_LEN = 64
TIMESTEP = 5

LABEL_IDX = 2

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def judger(x, threshold):
    if x > threshold:
        return 2
    elif x < -threshold:
        return 1
    else:
        return 0

class CNNBlock(nn.Module):
    def __init__(self, in_channels: int,
                       out_channels: int,
                       kernel_size: int,
                       dilation_size: int):

        super(CNNBlock, self).__init__()

        _L = SEQ_LEN
        _padding_size = (dilation_size * (kernel_size - 1)) >> 1

        _latent_size = (in_channels + out_channels) >> 1
        _latent_gru = math.floor(_latent_size * 7 / INPUT_SIZE)
        _latent_cnn = _latent_size - _latent_gru

        # 用于前7列
        self.gru1 = nn.GRU(input_size=7, hidden_size=_latent_gru)
        # self.sub_block1 = nn.Sequential(self.gru1, nn.ReLU(), self.gru2, nn.ReLU())
        # 用于8~end列
        self.cnn1 = weight_norm(nn.Conv1d(in_channels=in_channels - 7,
                                          out_channels=_latent_cnn,
                                          kernel_size=(kernel_size,),
                                          padding = (_padding_size,),
                                          dilation = (dilation_size,)))
        self.sub_block2 = nn.Sequential(self.cnn1, nn.ReLU())

        self.relu = nn.ReLU()

        self.cnn = weight_norm(nn.Conv1d(in_channels=_latent_size,
                                         out_channels=out_channels,
                                         kernel_size=(kernel_size,),
                                         padding = (_padding_size,),
                                         dilation = (dilation_size,)))


        self.resample = weight_norm(nn.Conv1d(in_channels,
                                              out_channels,
                                              kernel_size = (3,),
                                              padding = (1,)))\
            if in_channels != out_channels else None
        self.relu = nn.ReLU()

        self.init_weight()

    def init_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight.data, nonlinearity='conv1d')
                nn.init.zeros_(layer.bias.data)

    def forward(self, x: torch.Tensor):

        res = self.resample(x) if self.resample else x
        x1 = F.relu(self.gru1(x[:, :7, :].permute(2,0,1))[0])
        x2 = self.sub_block2(x[:, 7:, :])
        x = self.cnn(torch.cat((x1.permute(1,2,0), x2), dim=1)) + res
        x = self.relu(x)

        return x

class ConvLstmNet(nn.Module):
    def __init__(self, in_features, seq_len, out_features):
        super(ConvLstmNet, self).__init__()

        # set size
        self.in_features = in_features
        self.seq_len     = seq_len
        self.out_features = out_features

        _kernel_size = 3

        _layers = []
        _levels = [INPUT_SIZE, 96, 192, 128, 64, INPUT_SIZE]

        self.bn = nn.BatchNorm1d(num_features=in_features)

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

        self.featureExtractor = weight_norm(nn.Conv1d(in_channels=_levels[-1],
                                                      out_channels=3,
                                                      kernel_size=(62,)))


    def forward(self, x):

        x = self.network(x)
        x = self.featureExtractor(x)

        return x.permute(0,2,1)


def gen_open_pos(x, times = 1):
    # if x[1] == 2 or x[2] == 2:
    if x[LABEL_IDX] == 2:
        return 100 * times
    elif x[LABEL_IDX] == 1:
        return -100 * times

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
        model_path = '/home/yby/SGD-HFT-Intern/Projects/T0/CNN/train_dir_0/model/CNN/param_clsall_matrix_v3/'
        model_name = 'CNNLstmCLS_epoch_11_bs10000_sl64_ts3.pth.tar'
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
        data['cls_2'] = data['p_2'].apply(lambda x: judger(x, 0.0015))
        data['cls_5'] = data['p_5'].apply(lambda x: judger(x, 0.002))
        data['cls_18'] = data['p_18'].apply(lambda x: judger(x, 0.003))
        raw_data = self.remote_server.get_raw_bars(ticker = stock_id, date = date).set_index("time")

        np_data = data[factor_ret_cols].values.astype(np.float32)
        np_data = torch.from_numpy(np_data)

        data = data.reset_index()
        data = data.set_index('time')

        sp_dataset = HFDatasetCls(np_data, LEN_SEQ=SEQ_LEN, batch_size = 8000, time_step=1)

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

        p_2_pred = torch.cat(p2_res_list)
        p_5_pred = torch.cat(p5_res_list)
        p_18_pred = torch.cat(p18_res_list)


        p_2_pred_prob, p_2_pred = label_extractor(p_2_pred, prob_up = 0.7, prob_down = 0.6)
        p_5_pred_prob, p_5_pred = label_extractor(p_5_pred, prob_up = 0.7, prob_down = 0.7)
        p_18_pred_prob, p_18_pred = label_extractor(p_18_pred, prob_up = 0.7, prob_down = 0.7)

        pred_y = np.concatenate([p_2_pred, p_5_pred, p_18_pred, p_2_pred_prob, p_5_pred_prob, p_18_pred_prob], axis = 1)
        pred_y = pd.DataFrame(pred_y, index = data.index, columns = ['cls_2_pred', 'cls_5_pred','cls_18_pred',
                                                                     'cls_2_prob', 'cls_5_prob','cls_18_prob'])
        # pred_y = pd.DataFrame(pred_y, index = data.index, columns = ['cls_5_pred'])
        pred_y = pd.concat([pred_y, data[['cls_2','cls_5','cls_18', 'p_2', 'p_5', 'p_18']]], axis = 1)
        res = pred_y.merge(raw_data, how = "inner", left_index=True, right_index=True)


        res.reset_index(inplace = True)

        # res: pd.DataFrame = result[['date', 'time', 'server_time', 'local_time',
        #               'code', 'price', 'cls_2', 'cls_5', 'cls_18', 'cls_2_pred', 'cls_5_pred','cls_18_pred']]

        res.loc[:, 'predictions'] = list(res[pred_cols].values)

        times = 2 if stock_id.startswith('688') else 1
        res['Direction'] = res['predictions'].apply(lambda x: "Buy" if x[1] == 2 else ( "Sell" if x[1] == 1 else None))
        res['pos'] = res['predictions'].apply(lambda x: gen_open_pos(x, times = times))

        res['fillPos'] = res.pos.fillna(method='ffill')

        buy_close_condition = (res.fillPos > 0) & (res['predictions'].apply(lambda x: x[LABEL_IDX] == 0 or x[LABEL_IDX] == 1))
        sell_close_condition = (res.fillPos < 0) & (res['predictions'].apply(lambda x: x[LABEL_IDX] == 0 or x[LABEL_IDX] == 2))
        res.loc[(buy_close_condition | sell_close_condition), 'pos'] = 0
        res.pos = res.pos.fillna(method='ffill')
        res.loc[res['pos'] > 0, 'comm_signal'] = 0.0015
        res.loc[res['pos'] < 0, 'comm_signal'] = 0.0015
        res.loc[res['pos'] == 0, 'comm_signal'] = 0

        res['comm_price'] = round(res['last'] * (1 + res['comm_signal']), 2)
        res['comm_price2'] = res['last'] * (1 + res['comm_signal'])

        res.drop('predictions', axis = 1, inplace = True)

        for c in ['p_2', 'p_5', 'p_18', 'cls_2_prob', 'cls_5_prob','cls_18_prob']:
            res[c] = round(res[c], 4)
            res[c] = res[c].astype(str)
        res["custom_cls2"] = res.apply(lambda x: x['p_2'] + ":" + x['cls_2_prob'], axis = 1)
        res["custom_cls5"] = res.apply(lambda x: x['p_5'] + ":" + x['cls_5_prob'], axis = 1)
        res["custom_cls18"] = res.apply(lambda x: x['p_18'] + ":" + x['cls_18_prob'], axis = 1)

        res.rename(columns={'cls_2': "custom_cls2",
                            'cls_5': "custom_cls5",
                            'cls_18': "custom_cls18",
                            'cls_2_pred': "custom_cls2pred",
                            'cls_5_pred': "custom_cls5pred",
                            'cls_18_pred': "custom_cls18pred"}, inplace = True)

        return res



def query(stock_id, date):
    SAVING_PATH = '../Backtest/'
    if not os.path.exists(SAVING_PATH):
        os.makedirs(SAVING_PATH)
    predict = Predict()

    stock_bars:pd.DataFrame = predict.specific_stock(stock_id=stock_id, date=date)
    stock_bars['date'] = stock_bars.apply(lambda x: str(x['server_time'].date()), axis = 1)

    stock_signals = stock_bars[["date", "time", "code", "Direction"]]

    stock_bars.to_csv("%s%s_%s.csv" % (SAVING_PATH, stock_id, date), encoding = 'utf-8')
    stock_signals.to_csv("%s%s_%s_signals.csv" % (SAVING_PATH, stock_id, date), encoding = 'utf-8')



if __name__ == "__main__":
    query("688029", "20211105")