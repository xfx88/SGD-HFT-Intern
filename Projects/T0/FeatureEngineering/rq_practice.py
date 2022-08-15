from datetime import datetime
import rqdatac as rq
rq.init("15626436420", "vista2525")

start_date = 20210701
end_date = 20211030

# 获取交易日，比tushare更方便
rq.get_trading_dates(start_date=start_date, end_date = end_date)


# 获取上一个交易日的信息
rq.get_previous_trading_date(datetime.today())


# 获取距离指定日期最近的一次的指数权重，XSHE结尾是深交所的股票，XSHG结尾要么是上海交易所，要么是指数，看编号
rq.index_weights('000016.XSHG', '20160801')

# 获取市场信息，type=None返回全部金融工具
rq.all_instruments(type='Convertible', market='cn', date=None).head(20)

# 获取行业的全部股票
rq.get_industry('证券', source='citics', date=None, market='cn')

# 获取技术指标等因子数据
rq.get_factor(['000001.XSHE', '600000.XSHG'],'WorldQuant_alpha010', start_date = '20190601', end_date = '20190604')
rq.get_factor('000001.XSHE', ["a_share_market_val",'a_share_market_val_in_circulation'], start_date = start_date, end_date = end_date) # 获取市值



# 获取tick数据，frequency可以选择分钟、1d等等
previous_trading_day = rq.get_previous_trading_date(datetime.today())
rq.get_price(['000001.XSHE', '000009.XSHE'], previous_trading_day, previous_trading_day, frequency='1m', fields=['open', 'close'], expect_df=True).reset_index()
rq.get_price('000009.XSHE', start_date='2021-07-01', end_date='2021-07-01', frequency='tick', fields=None, adjust_type='pre', skip_suspended =False, market='cn', expect_df=True,time_slice=None)

# 获取一致预期
rq.get_consensus_comp_indicators('600000.XSHG','2021-03-01','2021-04-01',fields=['comp_con_eps_t1','comp_con_eps_ftm','ty_eps_t2'])