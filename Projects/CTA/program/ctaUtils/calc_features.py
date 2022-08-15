import numpy as np
import pandas as pd

# Function to calculate first WAP
def calc_wap1(df):
    wap = (df['bid_price1'] * df['ask_volume1'] + df['ask_price1'] * df['bid_volume1']) / (df['bid_volume1'] + df['ask_volume1'])
    return wap

# Function to calculate second WAP
def calc_wap2(df):
    wap = (df['bid_price2'] * df['ask_volume2'] + df['ask_price2'] * df['bid_volume2']) / (df['bid_volume2'] + df['ask_volume2'])
    return wap

def calc_wap3(df):
    wap = (df['bid_price1'] * df['bid_volume1'] + df['ask_price1'] * df['ask_volume1']) / (df['bid_volume1'] + df['ask_volume1'])
    return wap

def calc_wap4(df):
    wap = (df['bid_price2'] * df['bid_volume2'] + df['ask_price2'] * df['ask_volume2']) / (df['bid_volume2'] + df['ask_volume2'])
    return wap

# Function to calculate the log of the return
# Remember that logb(x / y) = logb(x) - logb(y)
def log_return(price_series):
    return np.log(price_series).diff()

# Calculate the realized volatility
def realized_volatility(log_wpr_ret_series):
    return np.sqrt(np.sum(log_wpr_ret_series**2))

# Function to count unique elements of a series
def count_unique(series):
    return len(np.unique(series))

# level-3 OBI and Depth Ratio
def weight_pecentage(weights, df):
    ask_name_list = ["ask_volume%d"%i for i in range(1, len(weights)+1)]
    bid_name_list = ["bid_volume%d"%i for i in range(1, len(weights)+1)]

    Weight_Ask = np.dot(df[ask_name_list], weights)
    Weight_Bid = np.dot(df[bid_name_list], weights)

    W_AB = Weight_Ask/Weight_Bid
    W_A_B = (Weight_Ask - Weight_Bid)/(Weight_Ask + Weight_Bid)
    return W_AB, W_A_B

# rise ratio
def calc_rise(log_price_series, lag_time:int):
    # make sure the index is replaced and sorted
    log_price_series.reset_index(inplace=True)
    conds = [(log_price_series.index < (lag_time)), (log_price_series.index >= (lag_time))]
    lag_log_ret = np.select(condlist=conds, choicelist=[log_price_series-log_price_series.iloc[0], log_price_series.diff(lag_time)], default=None)
    return lag_log_ret





