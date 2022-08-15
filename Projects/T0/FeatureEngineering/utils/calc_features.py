import numpy as np
import pandas as pd
import rqdatac as rq


rq.init("15626436420", "vista2525")

def calc_wap1(df):
    wap = (df['bid_price1'] * df['ask_volume1'] + df['ask_price1'] * df['bid_volume1']) / (df['bid_volume1'] + df['ask_volume1'])
    return wap

def calc_wap2(df):
    wap = (df['bid_price2'] * df['ask_volume2'] + df['ask_price2'] * df['bid_volume2']) / (df['bid_volume2'] + df['ask_volume2'])
    return wap

def calc_wap3(df):
    wap = (df['bid_price1'] * df['bid_volume1'] + df['ask_price1'] * df['ask_volume1']) / (df['bid_volume1'] + df['ask_volume1'])
    return wap

def calc_wap4(df):
    wap = (df['bid_price2'] * df['bid_volume2'] + df['ask_price2'] * df['ask_volume2']) / (df['bid_volume2'] + df['ask_volume2'])
    return wap

def calc_wap_all(df, level_range:list):
    """
    level_range: indicate which levels to include in the WAP process, eg. [1,2,3,4,5]
    """
    bp_name = ['bid_price%s'%i for i in level_range]
    ap_name = ['ask_price%s'%i for i in level_range]
    bv_name = ['bid_volume%s'%i for i in level_range]
    av_name = ['ask_volume%s'%i for i in level_range]

    # if using same sides to do the weighting
    wap = (np.sum(np.multiply(df[bp_name], df[bv_name]).to_numpy(), axis=1) + np.sum(np.multiply(df[ap_name], df[av_name]).to_numpy(), axis=1))/df[bv_name + av_name].sum(axis=1).to_numpy()
    
    # If using different sides to do the weighting 
    # wap = (np.sum(np.multiply(df[bp_name], df[av_name]).to_numpy(), axis=1) + np.sum(np.multiply(df[ap_name], df[bv_name]).to_numpy(), axis=1))/df[bv_name + av_name].sum(axis=1).to_numpy()
    return wap

def calc_vwap(df):
    df = df.copy()
    df['ttl'] = (df['high']+df['low']+df['last'])/3
    df['vwap'] = (df['ttl']*df['volume']).expanding(min_periods=1).sum()/df['volume'].expanding(min_periods=1).sum()
    return df['vwap']

def calc_spread(df):
    return (df['ask_price1'] - df['bid_price1'])/(df['ask_price1'] + df['bid_price1']) * 2 * 10000

def get_rq_factors(code_list, factor_list, start_date, end_date):
    """
    code_list: such as ['000001.XSHE', '600000.XSHG']
    factor_list: such as ["a_share_market_val",'a_share_market_val_in_circulation']
    """
    rq.get_factor(code_list, factor_list, start_date = start_date, end_date = end_date) 


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





