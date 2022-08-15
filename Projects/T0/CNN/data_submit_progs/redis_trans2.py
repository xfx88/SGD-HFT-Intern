import redis
# import utilities as ut
import pickle
from tqdm import tqdm
import os
from joblib import Parallel,delayed

start_date = 20210501
end_date = 20210630

rs = redis.Redis(host = "103.24.176.114", port = 6379, password="adGaqIwPFoDJ4Ljf")
redis_keys = list(rs.keys())
rs.close()
cnn_redis_keys = [x for x in redis_keys if 'CNN' in str(x)]
redis_keys = [x for x in cnn_redis_keys if (int(str(x).split('_')[1]) <= end_date)
                    and (int(str(x).split('_')[1]) >= start_date)]

file_path = "/home/yby/YBY/Data_56/"

def read_and_save_data(keys):
    rs1 = redis.Redis(host = "103.24.176.114", port = 6379, password="adGaqIwPFoDJ4Ljf")

    for key in tqdm(keys):
        df = rs1.get(key)
        df = pickle.loads(df)
        key = key.decode()
        df.to_pickle(f"{file_path}{key[4:]}.pkl")

    rs1.close()

if __name__ == '__main__':
    read_and_save_data(redis_keys)