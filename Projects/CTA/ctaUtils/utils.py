from difflib import restore
from multiprocessing.spawn import prepare
from typing import get_origin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from datetime import datetime
from datetime import timedelta
import os
import glob
import gc
import dask
import joblib
from multiprocessing import cpu_count
import functools
from itertools import product
from collections import OrderedDict, defaultdict, Counter
import warnings
import hfhd.hf as hf
from tqdm import tqdm, trange
import pickle
import redis
from ctaUtils.RemoteQuery import *
from ctaUtils.calc_features import *

warnings.filterwarnings("ignore")
pd.set_option('expand_frame_repr', False)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 180)
pd.set_option('display.max_columns', None)


class CTAUtils:
    def __init__(self):
        self.src = RemoteSrc()
        self.pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=False)
        self.server = redis.Redis(connection_pool=self.pool)

    