import pandas as pd
import numpy as np

model = TCN([4,37],4,[[12,12,12,12,12,12,12],
                        [111,111,111,111,111,111,111]],kernel_size=3,dropout=0.2)
model = model.to(device)
model = DDP(model,device_ids=[local_rank],
             #find_unused_parameters=True,
            output_device=local_rank)
model.load_state_dict(torch.load(f'tcn_reg{version}/{start_epoch}.pth',
          map_location=device))


model.eval()



def decode_fn(record_bytes):
    ds = tf.io.parse_single_example(
        # Data
        record_bytes,

        # Schema
        {"date": tf.io.FixedLenSequenceFeature((), dtype=tf.string, allow_missing=True),
         "data": tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
         "label1": tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
         "label2": tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
         "label3": tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
         "shape": tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True)
         }
    )
    date = ds['date']
    features = ds['data']
    shape = ds['shape']
    features = tf.reshape(features, [shape[0], shape[1]])
    #pad = tf.zeros([5000 - shape[0], 42])
    features1 = features[:, :2] #*1000
    features2 = features[:, 10:11]
    features3 = features[:, 13:43]/1000
    features4 = features[:, 43:49]/1000
    features5 = features[:, 51:53]
    features6 = features[:,53:54]
    features = tf.concat([features1, features2,features3, features4,features5,features6], 1)
    #features = tf.concat([pad, features], 0) #*1000
    #     features = tf.expand_dims(features,0)
    y = ds['label3']
    y = tf.reshape(y,(shape[0],4))
    #y_pad =tf.zeros([5000-shape[0],4])
    #y = tf.concat([y_pad,y],0)
    # y = y/tf.constant([0.00055,0.00081,0.0017, 0.0006])
    return date, tf.transpose(features, [1, 0]), tf.transpose(y, [1, 0])



buy_open_threshold = 0.002
sell_open_threshold = -0.002
buy_close_threshold = 0
sell_close_threshold = 0
comm_price_threshold = 0.001


def gen_open_pos(x):
    if all(x >= 0) and x.max() >= buy_open_threshold:
        return 100
    elif all(x <= 0) and  x.min() <= sell_open_threshold:
        return -100


codes = {'603290', '603893', '603260', '688029', '600563', '603444', '688099', '600556', '603345', '603605', '603806',
         '603486'}
# codes = {'688029'}
stock_path_list = os.listdir("/home/chenqisheng/data/500_wavenet_data_v8/")
test_path_list = []
for i in stock_path_list:
    if i[:6] in codes and int(i[7:15]) >= 20211101 and int(i[7:15]) < 20211201:
        test_path_list.append(i)
    else:
        pass

test_path_list = list(map(lambda x: "/home/chenqisheng/data/500_wavenet_data_v8" + "/" + x, test_path_list))

if os.path.exists(f"backtest_files{version}")==False:
    os.mkdir(f"backtest_files{version}")
if os.path.exists(f"backtest_files_concat{version}")==False:
    os.mkdir(f"backtest_files_concat{version}")


batch_size = 1
test_dataloader = TFRecordDataset(test_path_list)
for i, (date, images, labels) in enumerate(test_dataloader):
    d = list(map(lambda x: x.decode('UTF-8'), date))
    f = torch.Tensor(images)
    l = labels
    res = model(f.unsqueeze(0)).cpu().detach().numpy()[0]
    res = np.concatenate([l.T, res.T], 1)
    res = pd.DataFrame(res)
    res.columns = ['p_2', 'p_5', 'p_18', 'p_diff', 'p_2_pred', 'p_5_pred', 'p_18_pred', 'p_diff_pred']
    res['time'] = d
    res[['p_2', 'p_5', 'p_18', 'p_diff']] = res[['p_2', 'p_5', 'p_18', 'p_diff']].values
    res[['p_2_pred', 'p_5_pred',
         'p_18_pred', 'p_diff_pred']] = res[['p_2_pred', 'p_5_pred',
                                             'p_18_pred', 'p_diff_pred']].values * np.array(
        [0.00055, 0.00081, 0.0017, 0.0006])
    res[['p_2_pred', 'p_5_pred',
         'p_18_pred', 'p_diff_pred']] = res[['p_2_pred', 'p_5_pred',
                                             'p_18_pred', 'p_diff_pred']].values  # *np.array([2.4, 2.5, 4, 4])
    res['code'] = test_path_list[i].split("/")[-1][:6]
    res['date'] = test_path_list[i].split("/")[-1][7:15]

    dat = pd.read_csv("/sgd-data/t0_data/500factors/500factors/%s/%s.csv" % (test_path_list[i].split("/")[-1][:6],
                                                                             test_path_list[i].split("/")[-1][7:15]))
    res = res.merge(dat[['time', 'price']], on='time')
    res[['time', 'date', 'code', 'price', 'p_2', 'p_5', 'p_18', 'p_diff', 'p_2_pred', 'p_5_pred',
         'p_18_pred', 'p_diff_pred']].to_csv("backtest_files%s/%s_%s.csv" % (version,test_path_list[i].split("/")[-1][:6],
                                                                           test_path_list[i].split("/")[-1][7:15]),
                                             index=False)


def get_test_data(code,data_list,back_test_path,pred_col_name):
    # rs = ut.redis_connection()
    # cols = ['date','code','vwp','p_2','p_5','p_18','p_diff',pred_col_name]
    # keys = [x for x in test_redis_keys if code in str(x)]
    # data_list = [ut.read_data_from_redis(rs,key)[cols] for key in keys]
    for data in data_list:

        data['4_pred'] = list(data[pred_col_name].values)
        data['pos'] = data['4_pred'].apply(lambda x: gen_open_pos(x))
        if code.startswith('688'):
            data['pos'] = data['pos'] * 2
        data['fillPos'] = data.pos.fillna(method='ffill')

        buy_close_condition = (data.fillPos > 0) & (data['4_pred'].apply(lambda x: all(x<0)))
        sell_close_condition = (data.fillPos < 0) & (data['4_pred'].apply(lambda x: all(x>0)))
        data.loc[(buy_close_condition | sell_close_condition), 'pos'] = 0
        data.pos = data.pos.fillna(method='ffill')
        data.loc[data['pos'] > 0,'comm_signal'] = data['4_pred'].apply(lambda x : max(0, x[2] - comm_price_threshold))
        data.loc[data['pos'] < 0, 'comm_signal'] = data['4_pred'].apply(lambda x: min(0, x[2] + comm_price_threshold))
        data.loc[data['pos'] == 0, 'comm_signal'] = data['4_pred'].apply(
                                                    lambda x:  x[:3].sum())

        data['comm_price'] = round(data.price * (1 + data['comm_signal']), 2)
        data['comm_price2'] = data.price * (1 + data['comm_signal'])
    data = pd.concat(data_list,axis = 0)
    data['4_pred'] = data['4_pred'].apply(lambda x: list(x))
    data.to_csv(back_test_path + f'/{code}.csv')
    return data

codes = ['603290','603893','603260','688029','600563',
         '603444','688099','600556','603345','603605',
         '603806','603486']
#codes = ['688029']
stock_path_list=os.listdir(f"/home/chenqisheng/backtest_files{version}")
for i in codes:
    print(i)
    file_list=[]
    for j in stock_path_list:
        if  i in j:
            file_list.append(pd.read_csv(f"/home/chenqisheng/backtest_files{version}/"+j))
    data=get_test_data(i,file_list,f"/home/chenqisheng/backtest_files_concat{version}",[ 'p_2_pred', 'p_5_pred',
       'p_18_pred', 'p_diff_pred'])