from WindPy import *
import pandas as pd
import datetime as dt
import os

w.start()

from flask import Flask, render_template, request


# 获取ymd当天全部转债
def getAllCBList(ymd=dt.datetime.now().strftime("%Y%m%d")):
    # cbWData = w.wset("sectorconstituent", "date=" + ymd + ";sectorid=a101020600000000")
    # tmpDF = pd.DataFrame(cbWData.Data, index=None).T
    # tmpDF.columns = cbWData.Fields
    # 上交所
    shdata = w.wset("sectorconstituent", "date=" + ymd + ";sectorid=a101010206000000")
    tshDF = pd.DataFrame(shdata.Data, index=None).T
    # 深交所
    szdata = w.wset("sectorconstituent", "date=" + ymd + ";sectorid=a101010306000000")
    tszDF = pd.DataFrame(szdata.Data, index=None).T

    tdf = tshDF.append(tszDF, ignore_index=True)

    tdf.columns = ['date', 'wind_code', 'sec_name']
    tdf.reset_index(drop=True)

    return tdf


# 获取WSD的某个指标列表
def getWSD(codes, key, ymd=dt.datetime.now().strftime("%Y%m%d")):
    tmp = w.wsd(codes, key, ymd, ymd, "")
    return tmp.Data[0]


# 获取WSQ的某个指标列表
def getWSQ(codes, key):
    tmp = w.wsq(codes, key)
    return tmp.Data[0]


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    ymdNow = dt.datetime.now().strftime("%Y%m%d")
    ymdPost = ymdNow

    if request.method == 'POST':
        ymdPost = request.form.get('ymdDate')
    ymdFile = "cbCache/{ymd}.csv".format(ymd=ymdPost)
    if os.path.isfile(ymdFile):
        cbDF = pd.read_csv(ymdFile)
        cbCodeStr = ','.join(cbDF['wind_code'].values.tolist())
    else:
        cbDF = getAllCBList(ymdPost)


        cbCodeStr = ','.join(cbDF['wind_code'].values.tolist())
        # 正股code
        cbDF['underlyingcode'] = getWSD(cbCodeStr, 'underlyingcode')

        # 正股名称
        cbDF['underlyingname'] = getWSD(cbCodeStr, 'underlyingname')

        # 信用等级
        cbDF['creditrating'] = getWSD(cbCodeStr, 'creditrating')

        # 到期赎回价
        cbDF['maturitycallprice'] = getWSD(cbCodeStr, 'maturitycallprice')
        cbDF = cbDF[cbDF['maturitycallprice'].notnull()]
        cbCodeStr = ','.join(cbDF['wind_code'].values.tolist())

        # 到期日期
        cbDF['maturitydate'] = getWSD(cbCodeStr, 'maturitydate')
        cbDF = cbDF[cbDF['maturitydate'].notnull()]
        cbCodeStr = ','.join(cbDF['wind_code'].values.tolist())

        cbDF['maturitydate'] = cbDF['maturitydate'].apply(lambda x: x.strftime("%Y-%m-%d"))
        cbDF['date'] = cbDF['date'].apply(lambda x: x.strftime("%Y-%m-%d"))
        # 计算剩余天数
        cbDF['days'] = (cbDF['maturitydate'].apply(pd.to_datetime) - cbDF['date'].apply(pd.to_datetime)).apply(
            lambda x: x.days)

        cbDF = cbDF[cbDF.notnull()]
        cbDF = cbDF[cbDF.notna()]

        cbDF.to_csv(ymdFile, index=False)

    # 获取转债最新价
    if ymdPost == ymdNow:
        cbDF['rt_latest'] = getWSQ(cbCodeStr, 'rt_latest')
        cbDF = cbDF[cbDF['rt_latest'] > 0]
    else:
        cbDF['rt_latest'] = getWSD(cbCodeStr, 'close', ymdPost)
        cbDF = cbDF[cbDF['rt_latest'] > 0]

    # 计算简化利率
    cbDF['lv'] = (cbDF['maturitycallprice'] - cbDF['rt_latest']) / cbDF['days'] * 365

    cbDF = cbDF[cbDF['lv'] > 0]
    cbDF.sort_values(by=['lv', 'days'], ascending=[False, True], inplace=True)
    cbDF.reset_index(drop=True, inplace=True)
    cbDF.columns = '日期,转债代码,转债名,正股代码,正股名,信用等级,赎回价格,赎回日期,剩余天数,转债最新价,年化收益率'.split(',')
    dfHTML = cbDF.to_html()
    return render_template('index.html', dfHTML=dfHTML, ymdDate=ymdPost)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=55555, debug=True)
