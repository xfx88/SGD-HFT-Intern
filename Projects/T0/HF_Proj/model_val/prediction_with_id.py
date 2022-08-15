import datetime
from collections import namedtuple
import numpy as np
import pandas as pd

from typing import List
import torch
from torch.nn import functional as F
torch.device('cuda:1')

from src.dataset3 import HFDataset
from tst import TransformerEncoder
import tst.utilities as ut
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

RET_COLS = ['2', '5', '10', '20']

EgBar = namedtuple('EgBar', ['stock_id', 'date', 'TIME'])

# param2
# tOpt = ut.TrainingOptions(BATCH_SIZE=3,
#                           NUM_WORKERS=4,
#                           LR=1e-3,
#                           EPOCHS=120,
#                           N_stack=6,
#                           heads=4,
#                           query=32,
#                           value=32,
#                           d_model=128,
#                           d_input=43,
#                           d_output=4,
#                           )

tOpt = ut.TrainingOptions(BATCH_SIZE=3,
                          NUM_WORKERS=4,
                          LR=1e-3,
                          EPOCHS=80,
                          N_stack=7,
                          heads=4,
                          query=32,
                          value=32,
                          d_model=128,
                          d_input=42,
                          d_output=4,
                          )

def load_numpy_data(date_key, date_str):

    rs = ut.redis_connection(db = 0)
    data = ut.read_data_from_redis(rs, date_key)
    stock_index_dict = ut.read_data_from_redis(rs, b'stock_index_' + bytes(f'{date_str}', encoding='utf-8'))
    rs.close()
    return data, stock_index_dict

def load_df_data(eg: EgBar, seq_len = 50):
    query = f"df_{eg.date}_{eg.stock_id}"
    query = bytes(query, encoding = "utf-8")
    rs = ut.redis_connection(db = 0)
    df = ut.read_data_from_redis(rs, query)
    rs.close()
    df = df.loc[: eg.TIME]
    partition = df[ut.factor_ret_cols[1:-1]]
    df = df[RET_COLS]
    len_partition = len(partition)
    partition = torch.Tensor(partition.values).reshape(1, -1, 42)

    partition = partition[:, -seq_len :, ...]
    # elif len(partition) < seq_len:
    #     partition = F.pad(partition.transpose(1, 2), (seq_len - len_partition, 0), 'constant').transpose(1,2)

    return partition, len_partition, df


def generate_keys(start_date, end_date):
    rs = ut.redis_connection(db=0)
    all_redis_keys = rs.keys()
    rs.close()
    keys_of_dates = [x for x in all_redis_keys if (len(str(x).split('_')) == 2)
                     and (int(str(x).split('_')[1][:8]) <= end_date)
                     and (int(str(x).split('_')[1][:8]) >= start_date)]

    return keys_of_dates

class Predict:
    def __init__(self):
        self._load_model()
        self.model.eval()

    def _load_model(self):
        model_path = '../train_dir2/model/transformer/param_5/'
        model_name = 'ddpTransformer_epoch_20.pth.tar'
        model_data = torch.load(os.path.join(model_path, model_name))
        from collections import OrderedDict
        local_state_dict = OrderedDict()

        for k, v in model_data['state_dict'].items():
            name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
            local_state_dict[name] = v

        model = TransformerEncoder(tOpt)
        model.load_state_dict(local_state_dict)

        self.model = model.cuda()

    def all_bars(self, start_date, end_date):

        prediction_result_path = 'prediction/param_5/'

        all_keys = generate_keys(start_date, end_date)
        for date in all_keys:
            date_str = bytes.decode(date).split("_")[1]
            result_path = prediction_result_path + date_str + "/"
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            y_pred_all = []
            y_all = []
            test_data, stock_index_dict = load_numpy_data(date, date_str)
            test_dataset = HFDataset(test_data, LEN_SEQ=100)
            print(f"date {date}: prediction start...")
            with torch.no_grad():
                for idx, (x, y) in enumerate(test_dataset):
                    y_pred = self.model(x.cuda()).to('cpu')
                    y_pred_all.append(y_pred.detach().numpy())
                    y_all.append(y.detach().numpy())

                    if (idx + 1) % 100 == 0:
                        print(f"Tick {idx} is predicted.")
            y_pred_concat = np.concatenate(y_pred_all, axis=1)
            y_concat = np.concatenate(y_all, axis=1)
            date = str(date, encoding='utf-8')
            np.savez(f"prediction/param_2/concat_prediction_{date_str}.npz", y_pred=y_pred_concat, y_true=y_concat)

            for idx, (k, v) in enumerate(stock_index_dict.items()):
                individual_df = pd.DataFrame(data = y_pred_concat[idx, : len(v), :].reshape(-1, len(RET_COLS)),
                                             index = v,
                                             columns=RET_COLS)
                individual_df['code'] = k
                individual_df.to_pickle(result_path + f"{date_str}_{v}.pkl")

            print(f"date {date}: Prediction is done.")

    def specific_bar(self, egs: EgBar or List, seq_len = 20):

        if not isinstance(egs, list):
            egs = [egs]

        res = []
        rectify_std = np.array([0.00075, 0.001, 0.0013, 0.0017])
        for eg in egs:
            test_x, len_x, y_true = load_df_data(eg, seq_len)
            if not len(test_x.size()) == 3:
                raise AttributeError(f"Input must have shape of (batch_size, seq_len, features).\n")

            pred_y = self.model(test_x.cuda()).to('cpu')
            pred_y = pred_y.squeeze(0).detach().numpy()
            print(f"{eg.date} {eg.TIME} : ")
            print("prediction:", pred_y[-1])
            print("true value:", y_true.values[-1], "\n")
        # Plot
        encoder = self.model.layers_encoding[0]
        attn_map = encoder.attention_map[0].detach().cpu()
        plt.figure(figsize=(12, 12))
        sns.heatmap(attn_map)
        plt.savefig("attention_map")
            # if len_x < seq_len:
            #     pred_y = pred_y[-len_x:, ...]
        #     if len_x > seq_len:
        #         test_df = test_df.iloc[-seq_len :]
        #
        #     test_df[RET_COLS] = pred_y
        #     res.append(test_df)
        #
        # return res


if __name__ == "__main__":
    predict = Predict()
    # predict.all_bars(start_date=20211001, end_date=20211031)
    eg1 = EgBar('600418', '20211013', '09:30:20')
    # eg2 = EgBar('600418', '20210709', '09:30:47')
    # eg3 = EgBar('600418', '20210709', '09:36:59')
    #
    # eg4 = EgBar('600418', '20210723', '09:55:17')
    # eg5 = EgBar('600418', '20210723', '09:55:29')
    eg5 = EgBar('300274', '20211029', '09:36:15')
    eg6 = EgBar('601799', '20211018', '13:10:49')
    eg7 = EgBar('601799', '20211018', '10:35:11')
    eg8 = EgBar('601799', '20211018', '11:04:06')
    egs = [eg1, eg5, eg6, eg7, eg8]
    # predict.specific_bar(eg1)
    predict.specific_bar(egs)
