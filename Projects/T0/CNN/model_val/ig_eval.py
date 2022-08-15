import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import pandas as pd
import os
import numpy as np

import sys
sys.path.append("/home/wuzhihan/Projects/CNN/")

factor_ret_cols = ['timeidx','price','vwp','spread','tick_spread','ref_ind_0','ref_ind_1','ask_weight_14',
                   'ask_weight_13','ask_weight_12','ask_weight_11','ask_weight_10','ask_weight_9','ask_weight_8','ask_weight_7',
                   'ask_weight_6','ask_weight_5','ask_weight_4','ask_weight_3','ask_weight_2','ask_weight_1','ask_weight_0',
                   'bid_weight_0','bid_weight_1','bid_weight_2','bid_weight_3','bid_weight_4','bid_weight_5','bid_weight_6',
                   'bid_weight_7','bid_weight_8','bid_weight_9','bid_weight_10','bid_weight_11','bid_weight_12','bid_weight_13',
                   'bid_weight_14','ask_dec','bid_dec','ask_inc','bid_inc','ask_inc2','bid_inc2','turnover','p_2','p_5','p_18','p_diff']

INPUT_SIZE = 43
OUTPUT_SIZE = 4
SEQ_LEN = 32
TIMESTEP = 1

class ConvLstmNet(nn.Module):
    def __init__(self, in_features, seq_len, out_features):
        super(ConvLstmNet, self).__init__()
        # set size
        self.in_features = in_features
        self.seq_len     = seq_len
        self.out_features = out_features

        self.dropout = nn.Dropout(0.3)

        self.BN = nn.BatchNorm1d(num_features=in_features)

        self.conv_1 = nn.Conv1d(in_channels=in_features,
                                out_channels=32,
                                kernel_size=(3,),
                                padding=(1,),
                                bias=False)
        self.conv_2 = nn.Conv1d(in_channels=32,
                                out_channels=64,
                                kernel_size=(3,),
                                padding=(1, ),
                                bias=False)
        self.conv_3 = nn.Conv1d(in_channels=64,
                                    out_channels=128,
                                    kernel_size=(3,),
                                    padding=(1, ),
                                    bias=False)
        self.conv_4 = nn.Conv1d(in_channels=128,
                                out_channels=256,
                                kernel_size=(3,),
                                padding=(1,),
                                bias=False)
        self.conv_5 = nn.Conv1d(in_channels=256,
                                out_channels=512,
                                kernel_size=(3,),
                                padding=(1,),
                                bias=False)
        self.conv_6 = nn.Conv1d(in_channels=512,
                                out_channels=128,
                                kernel_size=(3,),
                                padding=(1,),
                                bias=False)
        self.conv_7 = nn.Conv1d(in_channels=128,
                                out_channels=64,
                                kernel_size=(3,),
                                padding=(1,),
                                bias=False)
        self.rnn = nn.LSTM(64, 32, num_layers=1)

        self.downsample = nn.Conv1d(in_channels=32,
                                    out_channels=4,
                                    kernel_size=(SEQ_LEN,),
                                    bias=False)

    def forward(self, x):
        x = self.BN(x)
        conv_feat = self.conv_1(x)
        conv_feat = self.conv_2(conv_feat)
        conv_feat = self.conv_3(conv_feat)

        conv_feat = self.dropout(conv_feat)

        conv_feat = self.conv_4(conv_feat)
        conv_feat = self.conv_5(conv_feat)

        conv_feat = self.dropout(conv_feat)

        conv_feat = self.conv_6(conv_feat)
        conv_feat = self.conv_7(conv_feat)

        conv_feat = conv_feat.permute(2,0,1)
        rnn_feat,_  = self.rnn(conv_feat)

        rnn_feat = self.dropout(rnn_feat)
        y = self.downsample(rnn_feat.permute(1,2,0)).squeeze(-1)
        return y

def load_model():

    model_path = '../train_dir_0/model/CNN/param_2/'
    # model_path = '../model/CNN/param_1/'
    model_name = 'CNN_epoch_30_bs20000_sl32_ts3.pth.tar'
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

    model = model.cuda()

    return model

def generate_input(stock_id, date, assigned_time, seq_len):
    # 加载 股票代码_日期 数据(time已是index)，取出指定时间帧
    sample_input = pd.read_pickle(f"/home/wuzhihan/Data/{stock_id}/{date}.pkl")
    sample_input = sample_input[factor_ret_cols[1:-4]]
    sample_input = sample_input.loc[:assigned_time]
    sample_input = sample_input.iloc[-seq_len :]

    sample_input = torch.tensor(sample_input.values.astype(np.float32), requires_grad=True)
    if len(sample_input) < SEQ_LEN:
        sample_input = F.pad(sample_input.transpose(1, 2), (SEQ_LEN - len(sample_input), 0), 'constant').transpose(1, 2)
    sample_input = sample_input.unsqueeze(0).permute(0, 2, 1)
    return sample_input

def calculate_ig(model, stock_id, date, assigned_time):
    ig = IntegratedGradients(model)

    sample_input = generate_input(stock_id, date, assigned_time, SEQ_LEN)
    sample_input = sample_input.cuda()

    # baseline = model(sample_input.cuda())

    return ig.attribute((sample_input, ),
                        # baselines=(baseline, ),
                        method='gausslegendre',
                        return_convergence_delta=True)

def main(stock_id, date, assigned_time):
    model = load_model()
    attributions, approximation_error = calculate_ig(model, stock_id, date, assigned_time)
    time_mean = attributions[0][0].mean(0).detach().cpu().numpy()
    factor_mean = attributions[0][0].mean(1).detach().cpu().numpy()
    time_mean = np.abs(time_mean)
    factor_mean = np.abs(factor_mean)

    time_importance = []
    factor_importance = []
    for i in range(10):
        idx1 = np.argmax(time_mean)
        time_mean[idx1] = 0
        time_importance.append(idx1)

        idx2 = np.argmax(factor_mean)
        factor_mean[idx2] = 0
        factor_importance.append(factor_ret_cols[idx2 + 1])


    print(factor_importance)
    print(time_importance)


if __name__ == "__main__":
    stock_id = '000009'
    date = '20211020'
    assigned_time = '10:45:18'
    main(stock_id, date, assigned_time)