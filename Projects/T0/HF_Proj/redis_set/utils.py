import redis
import os
import pandas as pd
import csv
import pickle
from joblib import Parallel, delayed

# def pickle_file(file_path, )

def io_submit_process(file_path, stock_ids):
    global_pool = redis.ConnectionPool(host='localhost', port=3056, db=0)
    r = redis.StrictRedis(connection_pool = global_pool)
    for stock_id in stock_ids:
        stock_path = file_path + stock_id + '/'
        dates = os.listdir(stock_path)
        dates.sort()
        for date in dates:
            date_path = f'{stock_path}/{date}'
            date_df = pd.read_csv(date_path)
            date_df['stock_id'] = stock_id
            df_name = f'{stock_id}_{date[:8]}'
            r.set(df_name, pickle.dumps(date_df))

if __name__ == '__main__':
    # file_path = '//sgd-data/t0_data/500factor/500factors/'
    # batch_num = 12
    # stock_list = os.listdir(file_path)
    # stock_lists = [stock_list[(batch_num * i): (batch_num * (i + 1))] for i in range(12)]
    # with Parallel(n_jobs=12) as parallel:  # 使用多进程模式
    #     results = parallel(delayed(io_submit_process)(file_path, stock_lists[list_id]) for list_id in range(len(stock_lists)))

    save_keys()