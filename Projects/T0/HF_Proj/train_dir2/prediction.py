import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
torch.device('cuda:1')
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from src.dataset import HFTestDataset
from tqdm import tqdm
import seaborn as sns

import gc
from tst import Transformer
import tst.utilities as ut

tOpt = ut.TrainingOptions(BATCH_SIZE=8,
                          NUM_WORKERS=4,
                          LR=1e-3,
                          EPOCHS=100,
                          N_stack=3,
                          heads=2,
                          query=8,
                          value=8,
                          d_model=128,
                          d_input=43,
                          d_output=1,
                          attention_size = 50,
                          window = 200,
                          padding = 50,
                          chunk_mode = 'window'
                          )

def predict():
    model_path = '../train_dir/model/transformer/param_comb_4/'
    model_name = 'ddpTransformer_epoch_60.pth.tar'