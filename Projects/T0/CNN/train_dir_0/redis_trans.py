import redis
import sys
sys.path.append("/home/wuzhihan/Projects/CNN/train_dir_0/")
import utilities as ut
import pickle
from tqdm import tqdm
import asyncio
import aredis


import os
from joblib import Parallel,delayed
start_date = 20210701
end_date = 20211130
# train_start_date  = 20210701
# train_end_date = 20211031
# test_start_date = 20211101
# test_end_date = 20211130
rs = ut.redis_connection()
redis_keys = list(rs.keys())
rs.close()
cnn_redis_keys = [x for x in redis_keys if 'CNN' in str(x)]
redis_keys = [x for x in cnn_redis_keys if (int(str(x).split('_')[1]) <= end_date)
                    and (int(str(x).split('_')[1]) >= start_date)]


# async def read_and_save_data(keys):
#     rs1 = aredis.StrictRedis(host = "103.24.176.114", port = 6379, password="adGaqIwPFoDJ4Ljf")
#     rs2 = aredis.StrictRedis(host = "127.0.0.1", port = 3056, password="")
#     for key in tqdm(keys):
#         df = await rs1.get(key)
#         key = key.decode()
#         await rs2.set(key, df)
#     rs1.close()
#     rs2.close()

def read_and_save_data(keys):
    rs1 = redis.Redis(host = "103.24.176.114", port = 6379, password="adGaqIwPFoDJ4Ljf")
    # rs2 = redis.Redis(host = "127.0.0.1", port = 3056, password="")
    # file_path = '/home/wuzhihan/Data/factors/'
    for key in tqdm(keys):
        df = rs1.get(key)
        key = key.decode()
        # rs2.set(key, df)
        # df.to_pickle(f"/home/wuzhihan/Data/factors/{key[4:]}.pkl")
        # res.append(df)
    rs1.close()

#
# def parallel_proc():
#     len_keys = int(len(train_redis_keys) / 4) + 1
#     train_redis_keyses = [train_redis_keys[len_keys * i : len_keys * (i + 1)] for i in range(4)]
#     Parallel(n_jobs=8, verbose=1, timeout=10000)(delayed(read_and_save_data)(k)
#                                                   for k in train_redis_keyses)


if __name__ == '__main__':
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(read_and_save_data(train_redis_keys))
    read_and_save_data(train_redis_keys[:10])