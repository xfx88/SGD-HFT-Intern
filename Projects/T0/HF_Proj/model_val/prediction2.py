import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
torch.device('cuda:1')
import torch.nn as nn
import os

from torch.utils.data import DataLoader, random_split
from src.dataset3 import HFDataset
from tqdm import tqdm
import seaborn as sns
from scipy.stats import spearmanr

import gc
from tst import TransformerEncoder
import tst.utilities as ut
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

RET_COLS = ['2', '5', '10', '20']

tOpt = ut.TrainingOptions(BATCH_SIZE=3,
                          NUM_WORKERS=4,
                          LR=1e-3,
                          EPOCHS=80,
                          N_stack=8,
                          heads=4,
                          query=32,
                          value=32,
                          d_model=256,
                          d_input=43,
                          d_output=4,
                          )


def load_data(date):

    rs = ut.redis_connection(db = 0)
    data = ut.read_data_from_redis(rs, date)
    rs.close()
    return data


def predict(model, date):
    prediction_result_path = '/prediction/param_2/'
    y_pred_all = []
    y_all = []

    test_data = load_data(date)
    test_dataset = HFDataset(test_data, LEN_SEQ=100)
    print(f"date {date}: prediction start...")
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_dataset):
            y_pred = model(x.cuda()).to('cpu')
            y_pred_all.append(y_pred.detach().numpy())
            y_all.append(y.detach().numpy())

            if (idx+1) % 100 == 0:
                print(f"Tick {idx} is predicted.")

    y_pred_concat = np.concatenate(y_pred_all, axis = 1)
    y_concat = np.concatenate(y_all, axis = 1)
    date = str(date, encoding = 'utf-8')
    np.savez(f"prediction/param_2/prediction_{date}.npz", y_pred = y_pred_concat, y_true = y_concat)

    # rs = ut.redis_connection(db=0)
    # stock_list = ut.read_data_from_redis(rs, b'stock_list' + b'_' + bytes(f'{date}'))
    # for i in range(len(stock_list)):
    #     stock = stock_list[i]
    #     df = ut.read_data_from_redis(rs, b'df' + b'_' + bytes(f'{date}_{stock}', encoding='utf-8'))
    #     df[RET_COLS] = y_pred_concat[i]

    print(f"date {date}: data is saved.")



def run_predict():
    predict_start_date = 20211001
    predict_end_date = 20211031
    rs = ut.redis_connection(db=0)
    all_redis_keys = rs.keys()
    rs.close()
    keys_of_dates = [x for x in all_redis_keys if (len(str(x).split('_')) == 2)
                    and (int(str(x).split('_')[1][:8]) <= predict_end_date)
                    and (int(str(x).split('_')[1][:8]) >= predict_start_date)]

    model_path = '../train_dir2/model/transformer/param_2/'
    model_name = 'ddpTransformer_epoch_79.pth.tar'
    model_data = torch.load(os.path.join(model_path, model_name))
    from collections import OrderedDict
    local_state_dict = OrderedDict()
    for k, v in model_data['state_dict'].items():
        name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
        local_state_dict[name] = v

    model = TransformerEncoder(tOpt)
    model.load_state_dict(local_state_dict)

    model = model.cuda()
    model.eval()

    for date in keys_of_dates:
        predict(model, date)



if __name__ == "__main__":
    run_predict()