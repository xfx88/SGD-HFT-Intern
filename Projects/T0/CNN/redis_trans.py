import train_dir_0.utilities as ut


train_start_date = 20210701
train_end_date = 20210930
test_start_date = 20211001
test_end_date = 20211031
rs = ut.redis_connection()
redis_keys = list(rs.keys())
rs.close()
cnn_redis_keys = [x for x in redis_keys if 'CNN' in str(x)]
train_redis_keys = [x for x in cnn_redis_keys if (int(str(x).split('_')[1]) <= train_end_date)
                    and (int(str(x).split('_')[1]) >= train_start_date)]
test_redis_keys = [x for x in cnn_redis_keys if (int(str(x).split('_')[1]) <= test_end_date)
                    and (int(str(x).split('_')[1]) >= test_start_date)]

assert 0 == 1