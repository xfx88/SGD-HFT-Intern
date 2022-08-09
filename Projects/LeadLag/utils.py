from difflib import restore
from multiprocessing.spawn import prepare
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from datetime import datetime
from datetime import timedelta
import dask
import joblib
from multiprocessing import cpu_count
import functools
from itertools import product
from collections import OrderedDict
import warnings
import hfhd.hf as hf
from tqdm import tqdm, trange

warnings.filterwarnings("ignore")
pd.set_option('expand_frame_repr', False)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 180)
pd.set_option('display.max_columns', None)

from RemoteQuery import *
src = RemoteSrc()

def get_stock_list(date_list):
    src = RemoteSrc()
    stock_list_final = []
    for date in date_list:
        code = src.get_stock_list(date)
        stock_list = pd.DataFrame(code, columns=['code'])
        # 为什么有重复代码
        stock_list.drop_duplicates(inplace=True)
        # 只取开头为000,60,002,30,43,83,87
        cond1 = (stock_list['code'].str.startswith('0')) & (stock_list['code'].str.contains('SZSE'))
        cond2 = (stock_list['code'].str.startswith('6')) & (stock_list['code'].str.contains('SSE'))
        cond3 = (stock_list['code'].str.startswith('30')) & (stock_list['code'].str.contains('SZSE'))
        stock_list = stock_list[cond1 | cond2 | cond3]
        stock_list['code'] = stock_list['code'].str[:6]
        stock_list = sorted(stock_list['code'].to_list())
        if stock_list_final.__len__() == 0:
            stock_list_final = stock_list
        else:
            # pick intersection
            stock_list_final = list(set(stock_list_final).intersection(set(stock_list)))
    print("Total number of stocks: %s"%stock_list_final.__len__())
    stock_list_final.sort()
    return stock_list_final

def normal_corr(x, y, l):
    if l > 0:
        # x leads y
        # calculate the correlation between current x and future y
        lag_y = y[l:]
        lag_x = x[:-l]
    elif l < 0:
        # y leads x
        # correlation between current y and future x
        lag_y = y[:l]
        lag_x = x[-l:]
    else:
        lag_x = x
        lag_y = y
    var = np.dot(lag_x, lag_y)
    corr = var/np.sqrt(np.sum(lag_x**2) * np.sum(lag_y**2))
    return corr

def prev_tick(df, namex, namey, l):
    return normal_corr(df[namex].to_numpy(), df[namey].to_numpy(), l)    


# HY estimator
def hy_corr(dfx, dfy, lag, choice='lead'):
    """
    calculate HY estimator using all the data all at once
    lag: should be a datetime.timedelta object
    lag > 0 means X leads Y
    """
    dfy = dfy.copy()
    dfx = dfx.copy()

    if (dfx.shape[0] < 50) | (dfy.shape[0] < 50):
        print("Data are not enough")
        return np.nan

    if choice == 'lead': 
        dfy['server_time'] -= lag 
        dfy['prev_time'] -= lag
        
    else: 
        dfy['server_time'] += lag
        dfy['prev_time'] += lag
    # num = dfx.apply(lambda x: (dfy[(x['prev_time'] < dfy['server_time'])&(dfy['prev_time'] < x['server_time'])]['delta_quote'] * x['delta_quote']).sum(), axis=1).sum()
    # deno = np.sqrt((dfx['delta_quote']**2).sum() * (dfy['delta_quote']**2).sum())
    # return num/deno
    hy = hf.hayashi_yoshida([dfx.set_index('server_time')['midquote'], dfy.set_index('server_time')['midquote']], choice='corr')
    return hy[0][1]

def hy_corr_mean(dfx, dfy, lag, date_list, choice='lead'):
    """
    calculate HY estimator on a daily basis and then take average
    return: the mean level of HY correlation (can also add 95% CI)
    """
    hy_corr_list = list(map(lambda date: hy_corr(dfx[dfx['server_time'].dt.date == pd.Timestamp(date)], dfy[dfy['server_time'].dt.date == pd.Timestamp(date)], lag, choice), date_list))
    corr_mean, corr_std = np.nanmean(hy_corr_list), np.nanstd(hy_corr_list)
    # return [corr_mean, corr_mean-1.96*corr_std, corr_mean+1.96*corr_std]
    return corr_mean


def LLR(dfx, dfy, grid):
    """
    grid: a positive grid of lag timedelta objects
    return: lead-lag ratio (LLR > 1 means X leads Y more significantly than Y leads X)
    """
    nums = np.array(list(map(lambda lag: hy_corr(dfx, dfy, lag, choice='lag'), grid)) + list(map(lambda lag: hy_corr(dfx, dfy, lag, choice='lead'), grid)))
    return sum(nums[nums.__len__()//2:]**2)/sum(nums[:nums.__len__()//2]**2)


def remove_zdt(dfx, dfy):
    """
    deal with limit-up and limit-down problem with HY estimator
    """
    dfx['limit'] = 0
    dfx['choice'] = 0
    dfy['limit'] = 0
    dfy['choice'] = 0

    # Attention!! The previous_time used for calculating HY estimator must be placed here to avoid large interval occurs
    # This process must be before deleting the data for limit-up/down and between different trading days
    # But if using the tick time, the process should be after deleting those non-tick time periods
    dfy['prev_time'] = dfy['server_time'].shift()
    dfx['prev_time'] = dfx['server_time'].shift()

    # remove corresponding data for y
    dfx.loc[dfx['delta_quote'] < -0.5, 'limit'] = 1
    dfx.loc[dfx['delta_quote'] > 0.5, 'limit'] = -1
    limit_time_list = dfx.loc[dfx['limit'] == 1, 'server_time'].to_list()
    finish_time_list = dfx.loc[dfx['limit'] == -1, 'server_time'].to_list()
    
    for limit_time in limit_time_list:
        end_time = pd.Timestamp(limit_time.date() + timedelta(days=1))
        # end_time = limit_time + timedelta(hours=8)
        for finish_time in finish_time_list:
            if (finish_time < end_time) & (finish_time > limit_time):
                end_time = finish_time
                break
        dfx.loc[(dfx['server_time'] >= limit_time) & (dfx['server_time'] <= end_time), 'choice'] = 1
        dfy.loc[(dfy['server_time'] >= limit_time) & (dfy['server_time'] <= end_time), 'choice'] = 1

    # remove corresponding data for x
    dfy.loc[dfy['delta_quote'] < -0.5, 'limit'] = 1
    dfy.loc[dfy['delta_quote'] > 0.5, 'limit'] = -1
    limit_time_list = dfy.loc[dfy['limit'] == 1, 'server_time'].to_list()
    finish_time_list = dfy.loc[dfy['limit'] == -1, 'server_time'].to_list()
    
    for limit_time in limit_time_list:
        end_time = pd.Timestamp(limit_time.date() + timedelta(days=1))
        # end_time = limit_time + timedelta(hours=8)
        for finish_time in finish_time_list:
            if (finish_time < end_time) & (finish_time > limit_time):
                end_time = finish_time
                break
        dfx.loc[(dfx['server_time'] >= limit_time) & (dfx['server_time'] <= end_time), 'choice'] = 1
        dfy.loc[(dfy['server_time'] >= limit_time) & (dfy['server_time'] <= end_time), 'choice'] = 1
    
    # remove corresponding data: 1. large midquote changes (when limit up or down occurs) 2. data during one asset in limit-up or down 3. data between the previous day and today
    dfx = dfx[(np.abs(dfx['delta_quote']) < 0.5) & (dfx['choice'] == 0) & (dfx['delta_quote'].notnull())]              
    dfy = dfy[(np.abs(dfy['delta_quote']) < 0.5) & (dfy['choice'] == 0) & (dfy['delta_quote'].notnull())]              

    return dfx, dfy

 

def obtain_cross_correlation(asset1, asset2, grid, date_list):
    dfx, dfy = asset1.copy(), asset2.copy()
    dfx, dfy = remove_zdt(dfx, dfy)

    # calculate all the HY correlations at once
    res_lag = list(map(lambda lag: hy_corr(dfx, dfy, lag, choice='lag'), grid))
    res_lead = list(map(lambda lag: hy_corr(dfx, dfy, lag, choice='lead'), grid))
    res_total = res_lag[::-1] + [hy_corr(dfx, dfy, timedelta(seconds=0))] + res_lead

    # calculate all the HY correlations on a daily basis
    # res_lag = list(map(lambda lag: hy_corr_mean(dfx, dfy, lag, date_list, choice='lag'), grid))
    # res_lead = list(map(lambda lag: hy_corr_mean(dfx, dfy, lag, date_list, choice='lead'), grid))
    # res_total = res_lag[::-1] + [hy_corr_mean(dfx, dfy, timedelta(seconds=0), date_list)] + res_lead    

    return np.array(res_lag), np.array(res_lead), np.array(res_total)
    

def process_one_stock(code, date) -> pd.DataFrame:
    asset = src.get_raw_bars(code, date)
    begin_time = '09:30:00'
    end_time = '14:57:00'
    range_ = (asset['time'] >= begin_time)&(asset['time'] < end_time)
    asset = asset[range_]
    # use log price
    asset['midquote'] = np.log((asset['ask_price1'] + asset['bid_price1'])/2)
    asset['delta_quote'] = asset['midquote'].diff()
    
    # use original time
    asset = asset[['code', 'delta_quote', 'server_time', 'midquote']]

    # use tick time only
    # cond1 = asset['delta_quote'] != 0
    # cond2 = np.abs(asset['delta_quote']) < 1
    # asset = asset.loc[cond1][['code', 'delta_quote', 'server_time']]

    return asset

def preprocess(code, date_list:list):
    try:
        res = list(map(lambda date: process_one_stock(code, date), date_list))
        asset_combined = pd.concat(res, ignore_index=True)
        asset_combined.sort_values(by=['server_time'], ascending=True, inplace=True)
        return asset_combined
    except:
        print("Fail to retrive the data for %s"%(code))
        return None
    

def prepare_all_data(code_list1:list, code_list2:list, date_list:list) -> pd.DataFrame:
    code_list = code_list1 + code_list2
    code_list = list(set(code_list))
    f_par = functools.partial(preprocess, date_list=date_list)
    df = joblib.Parallel(n_jobs=48)(joblib.delayed(f_par)(code) for code in code_list)
    # with dask.config.set(scheduler='processes', num_workers=48):
    #     result = dask.compute(dask.delayed(f_par)(code) for code in code_list)
    all_stock_data = pd.concat(df, ignore_index=True)
    all_stock_data.sort_values(['code', 'server_time'], inplace=True)
    return all_stock_data

def obtain_statistics(code_pair:tuple, stock_data, grid, date_list) -> OrderedDict:
    code1, code2 = code_pair
    df = stock_data.copy()
    try:
        asset1 = stock_data.loc[df['code'].str[:6] == code1, ]
        asset2 = stock_data.loc[df['code'].str[:6] == code2, ]
    except:
        print("Fail to retrive the data for the Pair (%s vs %s)"%(code1, code2))
        return None
    try:
        res_lag, res_lead, res_total = obtain_cross_correlation(asset1, asset2, grid, date_list)
        max_lead_corr, min_lead_corr, mean_lead_corr = res_lead.max(), res_lead.min(), res_lead.mean()
        max_lag_corr, min_lag_corr, mean_lag_corr = res_lag.max(), res_lag.min(), res_lag.mean()
        max_lead_corr, max_lag_corr = res_lead.max(), res_lag.max()
        min_lead_corr, min_lag_corr = res_lead.min(), res_lag.min()
        max_lead, max_lag = grid[res_lead.argmax()], grid[res_lag.argmax()]
        min_lead, min_lag = grid[res_lead.argmin()], grid[res_lag.argmin()]
        ori_corr = res_total[res_total.__len__()//2]
        llr = (res_lead**2).sum()/(res_lag**2).sum()
        print("maximum lead correlation of pair (%s vs %s) (lead = %s): %s"%(code1, code2, max_lead, max_lead_corr))
        print("minimum lead correlation of pair (%s vs %s) (lead = %s): %s"%(code1, code2, min_lead, min_lead_corr))
        print("maximum lag correlation of pair (%s vs %s) (lag = %s): %s"%(code1, code2, max_lag, max_lag_corr))
        print("minimum lag correlation of pair (%s vs %s) (lag = %s): %s"%(code1, code2, min_lag, min_lag_corr))
        print("mean lead correlation of pair (%s vs %s): %s"%(code1, code2, mean_lead_corr))
        print("mean lag correlation of pair (%s vs %s): %s"%(code1, code2, mean_lag_corr))
        print("Original correlation of pair (%s vs %s): %s"%(code1, code2, ori_corr))
        print("Lead/lag ratio of pair (%s vs %s): %s"%(code1, code2, llr))
        # summary = {'max_lead': max_lead, 'max_lead_corr': max_lead_corr, 'min_lead':min_lead, 'min_lead_corr': min_lead_corr, 'max_lag': max_lag, 'max_lag_corr': max_lag_corr, 'min_lag': min_lag, 'min_lag_corr': min_lag_corr, 'ori_corr': ori_corr, 'LLR':llr}
        # summary = {'max_lead': max_lead, 'max_lead_corr': max_lead_corr, 'min_lead':min_lead, 'min_lead_corr': min_lead_corr, 'max_lag': max_lag, 'max_lag_corr': max_lag_corr, 'min_lag': min_lag, 'min_lag_corr': min_lag_corr, 'ori_corr': ori_corr, 'hy_corr': res_total, 'LLR':llr}
        summary = OrderedDict([("max_lead", max_lead), ("max_lead_corr",max_lead_corr), ("min_lead", min_lead), ("min_lead_corr", min_lead_corr), ("max_lag", max_lag), ("max_lag_corr", max_lag_corr), ("min_lag", min_lag), ("min_lag_corr", min_lag_corr), ("ori_corr", ori_corr), ("hy_corr", res_total), ("LLR", llr)])
        return summary
    except:
        return None

def scan_all_pairs(code_list1, code_list2, date_list, grid) -> OrderedDict:
    all_stock_data = prepare_all_data(code_list1, code_list2, date_list)
    code_pairs = list(product(*[code_list1, code_list2]))
    f_par = functools.partial(obtain_statistics, stock_data=all_stock_data, grid=grid, date_list=date_list)
    result = joblib.Parallel(n_jobs=48)(joblib.delayed(f_par)(code_pair) for code_pair in code_pairs)
    # with dask.config.set(scheduler='processes', num_workers=cpu_count()-1):
        # result = dask.compute(dask.delayed(f_par)(code_pair) for code_pair in code_pairs)
    result = OrderedDict(zip(code_pairs, result))
    return result


def plot_corr(code1, code2, nums, result):
    index = [-i for i in nums[::-1]] + [0] + nums
    ratio = result['LLR']
    res = result['hy_corr']
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(10,5))
    plt.plot(index, res, color='b', label='LLR: %s'%ratio)
    # ax.set_ylim(min(res)-0.1, max(res)+0.2)
    # ax.set_ylim(-0.15, 0.8)
    ax.set_title('HY Correlation of %s vs %s'%(code1, code2))
    ax.set_xlabel('Lag')
    ax.set_ylabel('Cross-correlation')
    plt.legend(loc=0)
    plt.vlines(x = 0, ymin=-0.1, ymax=0.7, linestyles='dashed', color='r')