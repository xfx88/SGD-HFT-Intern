<font face="consolas">

<div align=center>

## 日内高频期货CTA

</div>

> ``Reference:``
> `Statistically Sound Machine Learning for Algorithmic Trading of Financial Instruments`

---

### 1. 期货分笔数据简介
1.熟悉软件、数据
2.简单的作图与统计操作
3.数理统计分析时间序列平稳性
4.对比两种回测方法（固定手数）
5.按固定金额回测
6.本周小结及作业

Babyquant提供7个商品期货的数据，都是黑色系，rb, i , j, hc, jm, SF, SM七个品种：rb是螺纹钢，i是铁矿石，j是焦炭，hc是热轧卷板，jm是焦煤，SF锰铁，SM硅铁。后续回测使用前五个。数据储存格式为pkl；为使程序快速运行，需电脑多核并行；这些品种都有夜盘，交易时间为白天的9:00-15:00，夜盘时间为21:00-23:00。

**期货的保证金（margin call）一般为12%（我们上3080总说一个东西叫追加保证金），会影响最终的收益率，但不会影响我们的资金曲线（比如收益率曲线是按照一张一张合约计算的，但是到期收益率取决于你初始的投入资金，因此会受到保证金的影响）**。

商品期货合约会到期，股票不会；对于发产品，需要现金管理，买逆回购等补贴收益，保证金低就可以有更多现金；股指期货持仓量单边计算，所以是单数，然而商品期货持仓量双边计算，都是偶数；**我们做量化交易更关注变化量而不是累积量，但交易所只发布累计量。**

什么是平均对手价买卖冲击成本？


商品期货的价格什么意思？例如螺纹钢rb价格是4000，表示的是4000元/吨，螺纹钢一手是10吨，10是合约乘数，因此一手螺纹钢合约价值约4000 * 10 = 4万元；螺纹钢最小跳价是1，因此报价一定是整数；交易一手螺纹钢保证金约10%，因此保证金大约需要4万 * 10% = 4000元；如果保证金不足，会被强制平仓（期货的高风险来源）；为了避免被强制平仓，一般使用保证金为总资金25-30%最好，例如账户有1万元，其中用2500-3000元用作保证金，其余闲置。

数据集是3天合并的，也就是每一个pickle都是三天合起来的，回测只用中间那天（有一列标签good=True）。例如日期20190808的数据，会有20190808和20190809的数据合并起来，但20190808的早上9:00:00到晚上夜盘结束的good这一列会等于True，也就是用来回测，其余good=False，是用来做预热和其它处理，不会用来回测，因此，某天日期的数据，可以认为是该天9点开始作为交易回测的起点。3天合并因为第一天作为预热，比如后面计算因子会用到EWMA，指数移动均线，前面的均线数值不稳定，值不准确，需要前一天的数据预热（因为商品or股票刚上市的时候价格都会出现猛涨以及大波动等情况，而且比如20ma在前20天的数据量都不够）。这样有了一定长度的预热数据后，可以用历史数据来预热，当天开盘时候的计算的因子已经稳定，可以直接拿来交易，避免开盘时因子计算不准确。**我们回测用的成交价（trade price）是下一跳的对手价，对于小资金来说最准确（why？后面会解释）** 每个产品都是1067天的数据。第三天用来计算因变量，比如未来的收益率，因此需要后面那天（我个人感觉babyquant这么合并数据就是为了方便regression或者怎样的，每个小的pkl文件都有三天，就不用去做一些shift，搞一些时间上的平移）。

使用数据总结：
- 样本内数据日期为5个产品1067天的数据，从2017.01.03到2021.05.25。
- 样本外为5个产品从2021.05.25到2021.08.17的数据。

当然，最大的好处就是，我们做这种高频的CTA，相当于因子计算只用了一两天的数据，没有说rolling个几十天，对于每一天的回测，我们都只用了2、3天的数据，**而实盘的时候我们也只需要前2、3天的数据就能计算因子，然后去执行交易操作，这样就可以保证回测与实盘的计算一致，且可以大大减少实盘的数据量**,因为我们都知道实盘是要跑的越快越好。如果说你每天的回测，计算个因子都要往前半年的数据，那你实盘也需要半年的数据，基本就寄了。**如果你计算因子是滑动用半年一年的，那如果是非超高频的分钟级别、小时、日度那OK，但是这里是分笔tick行情，那样计算且不说花费的时间，内存都要爆炸。因此使用2、3天的数据拼接进行回测是最好的选择。**

中国期货交易所只**公布了总的成交量，因此需要自己估计主动买入和主动卖出的成交量**（也就是交易所的trade的data里面没有BS标志，没有那个1和-1的标签）。因此可以看到数据里面有一档主动买量、卖量和其他档位的主卖主买的量。怎么估计的，我自己知道一个方法，算法交易课讲过，当然应该这里用的是其他的方法，因为那个方法无法区分档位。每笔交易都有一方买一方卖，但先挂单的是被动，后来的是主动，不存在同时。数据字段里buy.trade/sell.trade默认是在一档价格成交的主买主卖量。字段buy2.trade/sell2.trade其他档位成交的主买主卖量。

这个课程叫做机器学习做日内高频CTA，这个日代表一个自然日（早上9点到夜盘结束），而不是一个交易日，因为商品期货是有夜盘的，这样更符合我们的习惯，因为交易所是晚上9点开盘，不太符合日常习惯。

我们一般说做高频做CTA最多吃的就是1.手续费，2.价差。当然最多最多的还是spread，**因为在偏高频的交易中，买卖价差占据了绝大多数的交易成本**。我们开发高频策略，当然是不能抱有任何的侥幸心理，要保证策略的利润是能够严格cover掉手续费+spread的。因此后续策略开发都默认在ask1买入，bid1卖出，也就是每次交易会亏一个买卖价差，策略的回测更加严谨。而既然如此，我们下的就是**限价单**，**而我们限价单使用的不应该是当前这时刻的bid1和ask1，而应该使用那个trade price字段**。为什么呢？前面说我们回测数据里面那列当前时刻的trade price用的是下一跳的对手价，里面还有next.bid/next.ask是下一跳的价格。也就相当于我们回测的时候下单使用的是未来信息，比如卖，bid价未必能卖出，但next.bid一定能卖出；比如买，ask价未必能买到，但next.ask一定能买到。<font color=blue>虽然用的是未来信息，但这也是最保守的回测方式，比单纯加滑点要准确（我们平常都是在买一、卖一之类的加一个滑点0.01之类的，不够严谨），避免出现回测赚钱实盘不赚钱的情况</font>。要知道我们平常用什么滑点slippage去估计真正的成交价，反而没这种严谨，因此使用next.bid/next.ask用来回测最保险，因为是确定能够成交的价格。

使用一个limit order book的例子来说：
例如当前盘口200@4001-50@4002（4001是200手，4002是50手），表示在价格4001有200手买单，在价格4002有50手卖单。如果下一跳10@4002-500@4003，说明价格涨了，但我们直接在4002未必能买到，如果买不到我们的单子就会属于10@4002中的一个。但我们用next.ask，也就是4003，一定可以买到。所以这就是使用下一跳价格的原因。

WPR: 加权平均价。经典字段，kaggle volatility比赛里面都需要用到的feature，同时也是计算realized volatility的字段。
$$
wpr = \frac{bid.price\times ask.qty+ask.price\times bid.qty}{bid.qty+ask.qty}
$$
**因为卖量越多，价格越容易跌，均衡价格越接近买价，所以它是买价的权重；买量越多，价格越容易升，均衡价格越接近卖价，所以它是卖价的权重。同时注意额外处理涨跌停，涨停时ask_price1 = 0，跌停时bid_price1 = 0。**

#### 1.3 时序平稳性检验
一个ADF检验，augmented Dickey-Fuller Test，一个KPSS检验。
- ADF：p值越小越平稳，拒绝原假设就是平稳
- KPSS：p值越大越平稳，接受原假设就是平稳
- 一般来说adf更容易平稳，kpss稍微难一点
- 一般来说，时间间隔越短越平稳，因为变化小

我们要建立回归模型，当然希望数据是平稳的。而且当然数据采样频率越高，间隔越短越平稳。但是比如0.5s的tick是平稳的，这个很容易成立，但是也基本没什么用，因为时间间隔太短，平稳的你预测出来也无法在这么短的时间内盈利。所以这个时序检验要逐步增大时间间隔，比如间隔120tick，约一分钟，或者其他的频率。当然间隔越大，越不容易平稳，预测的难度也会越高，如果能预测盈利那可实操性也会更强。代码里面隔了2000跳，取了数据，竟然还是平稳的。说明螺纹钢的平稳性还挺好。然后由于每隔2000跳采样一天数据太少，所以程序里面写了个对日期并行的采样函数，整体也就1min就采样好了。

我们还是希望尽可能的平稳，这样历史发生的情况未来才可能发生，才可能进行预测，不然再好的模型效果也不行。哪怕是使用一些波动率过滤也要保证平稳。如果不平稳，比如过去可以取很大的值，未来可能就取不到，这样过去赚钱的未来可能就赚不到。如果时间序列平稳，那我们的回归就不是伪回归；我们希望平稳，是因为这样的话过去的事情可以更容易重复发生。

不平稳数据对因子计算、样本内外回测的影响：
- 例如异方差，过去波动大，现在波动小，则样本外可能会缺乏交易次数
- 如果过去波动小，现在波动大，则样本外可能频繁交易而亏钱
- 因此我们希望因子有平稳性，过去的取值范围跟现在差不多
- 另外，如果因子取值过于平均，则很难触发交易，我们也不希望这样

#### 1.4 两种回测方法
- ##### 第一种：适合低频交易
我们基于历史数据预测收益率return，然后将Return的预测值跟阈值threshold对比。如果return大于阈值，则做多，如果return小于负的阈值，则做空，预测收益率在正负阈值之间，则仓位为0，也就是平仓。**进一步的改进是根据预测的return的大小，来决定买入或卖出的仓位**。这样可以让我们的策略更加灵活，更加适合低频交易， 而且可以更好的控制交易风险。同时滑点、手续费影响不大（要么是按固定比例收手续费，要么就是固定值）；持仓时间本来就可以比较长，比如日线级别的策略，同时需要考虑涨跌停板不能买卖的情况。

优点：1.速度可以很快，因为只涉及向量计算，没有循环，不存在路径依赖
手续费有些按比例，有些固定；2.Threshold调整频率，threshold越小频率越高，可以通过阈值调整开平仓频率。

缺点：尽管可以通过阈值控制频率，但是也很容易造成频率可能过高，哪怕使用了未来数据也稳定亏钱；平均利润是买卖价差的倍数，最终会产生大量的亏损。因为预测值稍有变化仓位就要改变，导致频率过高，持仓时间过短，无法覆盖成本。而如果是日线低频，数据波动幅度远超手续费，可能就没有关系。多因子选股之类的适合这种回测方法（中低频）。


- ##### 第二种：适合高频交易
如果return大于阈值，则做多，signal=1, pos=1，如果return小于负的阈值，则做空, signal=-1, pos=-1，预测收益率在正负阈值之间，signal是0，但position不为零，仅仅是维持原来仓位，这样可以降低一些噪音。但是路径依赖，无法向量化计算。如果预测收益率大于阈值，signal设置为1，小于负的阈值，signal设置为-1，否则signal设置为0。如果signal为1，则position为1，如果signal为-1，则position为-1，否则position维持<font color=blue>原先仓位不变</font>。

根据这里的叙述，似乎我们没有涉及到做空，pos=1是做多，pos=-1是平仓？不不。我们高频和中低频不一样的，别忘了中低频alpha里面，有信号我们就直接全仓，然后信号变-1了我们就T+1或者T+0直接平仓。但是这里我们是高频，是固定手数回测，只要信号是1我们就加一手，也就意为着可能一直加，也可能一直平直到做空，反正无论怎么发展，我们都会在日度结束的时候平仓。

我们在回测的时候会加上一些噪音noise，同时使用threshold去过滤噪音。


- ##### 注意
上面这两种回测方法，都是**固定手数回测**，也就是每次开仓平仓都是一手这样去回测的，pos=1就买入一手这样。代码里还给了**固定金额回测**，核心代码改了，这就是固定比如10000快钱，每次开仓平仓都是10000这样去回测的，pos=1就买入10000这样，然后倒推买卖的手数，**邢大的之前代码框架都是按照固定金额回测的**。虽然我们平常买习惯比如按手去买，但是回测的时候还是会使用固定金额回测，原因很简单啊，就是比如你固定10000进去，你就可以统计资金曲线，你就对整体的收益率、回撤这些就有清晰的认知，同时可以不同策略、不同品种横向对比。然而如果你是固定手数回测，每次开仓平仓买入一手卖出一手，那这样你得到的净值曲线，包括各种策略评价指标很难和其他品种比较，不同产品的手数衡量不一样，数字上也没有直观的含义，因为那样曲线最后的净值不是你的收益率，你没办法可视化了解这个收益率。

整体来说固定手数回测和固定金额回测，核心逻辑代码一样，就在中间部分计算手数和金额的时候有修改，那开平仓的方法仍然是那两种，固定金额回测只是把钱拆分成多少手，固定手数只是把手数拆分成多少钱。

但是问题是，为什么PPT里面那个固定金额回测的代码在课件代码里面没有？？什么TMD情况？？TMD又是视频里面的代码有，但是拿到的代码没有。真实日了，自己找时间补上。应该是视频里面最后的一部分，拿到的代码没有。


其他：
- `position_pos.ffill(inplace=True)`**可以大幅度提高回测速度，why？**
- `position[n_bar-1] = 0`和`position[n_bar-2] = 0`，**因为我们做的是日内交易，需要在每个bar结束时，清空仓位，也就是收盘前平仓**。


---

### 2. 构造第一个因子
策略实盘一定要用C++，甚至linux的C++都会比windows的C++快很多，much less python。只是策略开发研究，以及回测是用python。

为什么要计算wpr的对数收益率？对数收益率有什么好处？方便计算,对数收益率是**多空对称**的，从x变化到y，以及从y变化到x对称的，log(80)-log(100)=-(log(100)-log(80))。

数据时间是三天，前面预热，中间回测，后面因变量，早盘9:00-15:00，晚盘21:00-23:00。期货交易所给的数据，每一小节的结束时间可能是多出一个500毫秒，比如铜的夜盘在01:00:00.500结束，如果清洗数据，很多人直接用01:00:00截取，就会丢掉后面的行情，而如果用01:00:01来截取，这样对上期所不会出错。babyquant自己分析2010-2020这10年的数据，最奇怪的会多出1.x秒，所以用02秒结束来截取是安全的，但不排除未来有奇葩行情，也可能交易所删除了这500毫秒。

``def zero_divide(x, y)``函数的作用是防止分母是零出错，此时对应数值返回零。

三个因子：
- ``factor_total_trade_imb_period``: 累计成交量的买卖不平衡
- ``factor_trade_imb_period``: 买卖不平衡量的移动均线
- ``get_atr(file, product, period_list, spread)``: 震动幅度
  
前两个是方向性因子，最后一个是波动性因子。atr是absolute true range，也就是振幅的意思。

主动买卖量不平衡显然是一个非常经典的因子，我们希望通过主动买卖量的差别来衡量当前的市场状况。total_trade_imb：计算（买量-卖量）的移动均线，再除以总成交量的移动均线；trade_imb：先计算（买量-卖量）/（买量+卖量）的比值，再计算比值的移动均线。哪种更好？**就是加减乘除计算和移动均线的先后顺序，哪个更好？**<font color=blue>结论是对于商品期货trade_imb会更好一些，也就是先加减乘除再计算均线会更好。但是两者表现好的时间段和品种都不大一样，因此也算是不同类型的因子</font>。

这里构造的波动率因子，就是振幅，atr，absolute true range，把振幅除了一个spread。为什么这么做？在偏高频的交易中，买卖价差占据了绝大多数的交易成本，我们可以用atr/wpr来对atr进行标准化，也就是振幅除以WPR来做标准化。<font color=blue>这样得出的波动率是百分比，因此跨品种对比更有意义，这个比值才能衡量品种做趋势策略的盈利难易程度，比值越大说明越容易通过趋势策略盈利，越小说明越难通过趋势策略盈利</font>。**上次是使用了atr/spread，也就是振幅除以价差，毕竟价差spread是最典型的波动性因子（本身价差大小不具备趋势预测），然而现在觉得不大好，特别对股指等小价差品种，价差过小会导致这个商的结果很大，没有起到标准化的效果。**

我们这里区分一下**方向性因子**和**波动性因子**。方向性因子就是可以进行收益率预测的因子，比如市值因子、反转因子，或者MACD，因子值的大小或者正负就是市场的预测。而波动率因子是没有方向的，只是提供一个指标，可以用来判断市场的走势，但是不能进行收益率预测，比如已实现波动率，并不能说值大了就要涨小了就跌，所以是无方向的，只能提供信息。

因子构造完成后需要做统计分析，做一些类似标准化等的操作，让范围比较小，同时分析下因子值的IC，IR，时序平稳性等等。重点还是平稳性，要除掉一些outlier，具体的操作方法我还不是特别有经验。

**为什么算出来的因子希望时序是平稳的？按照babyquant的说法，如果因子不平稳，需要做额外处理，比如什么波动率过滤等等。为什么需要这么做呢？**结果发现是hc的第二个因子不是很平稳，然后把因子的时序图画出来发现，就是有几个outlier导致的，那么去除掉一些outlier和极值，做一些过滤就是能用的了。后续结果发现，不是很平稳的话，全局优化grid后的回测结果不是很好，果然发现hc的那个第二个因子不平稳，回测效果不好。


方向性因子当然好，**可以直接用来交易**。方向性因子和波动性因子本身比较好判断，**但是我们往往无法判断是正向还是负向**。比如这个主动买卖量不平衡，如果是顺势交易，意味着买量强于卖量就做多，如果是逆势交易，意味着买量弱于卖量就做多。但我们不知道这因子是顺势还是逆势。因此，单因子分析只能是顺势和逆势都回测一遍，看哪个更好。在代码里面，我们专门有参数**reverse=1/-1来表明是顺势还是逆势，往往都需要试一遍**。如果回测结果表明逆势明显优于顺势，那么因子可以理解为逆势的，**然后再从经济学上找具体的原因。一般不可能两边都表现好，但有可能两边表现都不理想**，**那只能舍弃这个因子**。也可能绝对值小的时候顺势，绝对值大的时候逆势。但日内高频因子可能不会出现，中低频更容易出现这种因子。

方向性因子是好的，但是单纯使用方向性因子也有弊端，需要加入波动率因子。原因就是例如买卖量不平衡，**无论低波动还是高波动，比值都是-1到1之间，但在低波动的时候可能完全无法盈利，就是值一直在很小的范围内，始终没有预测出信号**，怎么办呢？**此时就需要波动率因子，让合成的因子整体在高波动和低波动的行情都能赚钱，都有信号产生**。



#### 2.3 因子回测
测试步骤：
1. 计算因子取值的分布，all_signal
2. 选好20个开仓阈值：open_thre
3. 每个开仓阈值平均分配5个平仓阈值，一共100组参数
4. 并行优化，面板优化。

这个意思就是设置一个开仓阈值（也就是signal=1）的grid，然后每个开仓阈值，设置一个平仓（也就是signal=-1）阈值的grid，就是取开仓阈值的一个百分比的grid。**然后这个开仓阈值和平仓阈值就是我们要优化的超参数**。这里采用的就是**面板优化grid search超参数**。我们还学过random search，也可以进一步优化，当然还有**更好的贝叶斯超参数优化**，找时间看一看python怎么实现，未来考虑，更加高效。

**下面说的全局优化，意思就是在样本总体上优化的这些超参数。**

```{r}
f_par = functools.partial(get_signal_pnl, product=product, signal_name=signal_name, thre_mat=thre_mat,
reverse=1, tranct=tranct, max_spread=spread*1.1, tranct_ratio=True, atr_filter=0.01)
```
1. 筛选策略：``good_strat = trade_stat["final.result"]["avg.pnl"]>2*spread``即至少跨越两倍的spread。


- ##### 因子优化方法介绍
目前单纯是使用的是**网格优化**的办法，同时我们并不知道因子是趋势还是反转，所以都要测测；reverse=1表示趋势，reverse=-1表示反转；未来纳入回归模型就可以由系数自动调整，也就是自动因子优化，现在还不行。可以未来考虑**贝叶斯优化**的进步。

- ##### 波动率过滤
加入atr过滤低波动率，波动太低可能无法覆盖成本，这里是使用atr>=0.01，也就是波动幅度要达到1%。当然，加入后可能交易信号会滞后。


#### 2.4 样本内外的设置
刚刚代码做了全局优化，可能说服力不强。现在介绍样本内外优化，构造训练集和测试集划分，例如训练集：2018年4月以前，测试集：2018年4月以后。我们需要**兼顾单笔利润和交易次数**，一般来说筛选策略的标准不宜太苛刻，否则容易过度优化，时间序列一般用前面训练，后面预测，反过来意义不大。

结果的评价方式：如果全局好但样本内外不好，则可能样本内过度优化。如果全局也不是很好，那么就是模型不适合行情。如果全局优化的时候，样本外这部分也不是很好，那么现在分样本内外，样本外这部分也大概率不好。这更多是行情的问题，而不是过度优化，因为只有一个因子，过度优化的可能性不是很高。

额外可以自己单独尝试不同的训练集和测试集划分，同时使用period=1024/2048，课件内只使用了4096。

#### 2.5 结论
基本说这个因子是商品期货里面**表现最好的因子**，一个因子也是可以玩出很多花样的，原因说在C++课里面就会知道。这些东西只有做高频的人才会知道的。


---

### 3. 生成多个新因子
#### 3.1 答疑
预热数据要使用多少？
  我们合成数据只是为了以备不时之需，没有说会用到所有的数据做预热。比如你算EWMA，长度是20，那前面1到19就可能有问题，rolling长度为4096，那有问题的数据就会比较多，所以加一天的数据来预热。同时注意，在EWMA函数中，自定义的加入了一个参数叫adjust，adjust=True可以使得因子尽快收敛，取值变正常（不会取值幅度过大），避免了随机交易，这样对预热的长度就不会太敏感，因为收敛的快。

EWMA周期为什么选择4096？
这个4096是ewm的半衰期参数halflife，然后取mean。
这个关系不会很大，一般会有最低限度，大家可以尝试2048、1024等等；只是**经验来说，商品期货取4096比较好**；股指波动更大，可能取小一点也好，不过看4096效果也不错；**而国债波动比较小，取大一些的数值好一些**。这个是**波动率上的考虑**，

另外特别注意，如果使用的是中低频数据比如用5分钟K线，不是tick数据，相当于5\*60\*2=600ticks，600个500毫秒，我们用4096相当于回看**半小时左右**。但是如果是但是如果用5分钟的K线，只有6、7个K线，随机性太强，不大好，行情可能在5分钟内走完了，交易点不大好；如果改用4096作为回看，逐点触发，那么可以在较好价位成交。**什么意思？**这个应该是说，如果数据是分钟级别，回看半小时，**用500毫秒的tick数据，一定比用分钟K线数据更准确，而且策略容量也会大一些，因为可以交易的时间也会变多**。


**1个开仓阈值对应5个平仓阈值，开仓和平仓阈值如何决定？**
- 我们上周程序对开仓平仓阈值做了网格优化。具体这个值怎么定的呢？那个网格。目前是一个具体值，不是很好，因为你不同品种肯定是需要分别设置分别调参的。所以应该使用分位数。

- <font color=blue>未来开仓值应该由分位数决定而不是具体值大小，一个是避免每个品种分别调参，另一个是避免过拟合；平仓值影响不大，只是怕开仓值过大的时候策略难以平仓导致风险太大；如果开仓值适中，基本上策略非多即空就可以了，两个参数反而容易过度拟合；中低频可能更需要平仓值，因为没有了日内的束缚，策略持仓时间可能很长，特别一些反转的策略，越是亏钱，越不会平仓；日内就不怕，因为当天结束就会强平，有固定的时间限制也就是日内结束会平仓，不会越亏越多。</font>
- 就是说，如果是中低频，我们之前做中低频的动量和反转，那尤其是反转，就要注意一下平仓的阈值，因为反转策略可能越跌越认为会反转，可能会让你亏死，所以一定要注意好平仓的阈值，或者做好止损条件。但是日内没关系，反正当天结束就会强平，所以策略非多即空即可，


**收益归一化为什么用指数移动平均线/绝对值？归一化收益大于分位数为什么可以作为开仓信号？**

- 这个对应的是``normalized return (nr)``因子，这节课介绍，归一化让取值幅度更统一，这样阈值更稳定；类似机器学习、神经网络输入进去每一步都要对因子重新标准化一样；
- 收益大于分位数说明处于极端状态，市场不够有效，提供了交易机会；
- 如果市场稳定，那么因子取值也是零附近，不需要交易，甚至可以考虑平仓。

**next.bid, next.ask怎么算？**
- next.bid/next.ask就是**下一跳的价格**，保证能够成交的价格，用来回测最保险。当然，更高频率（超高频交易）的不能用这种回测方法，只能用**对手价（主动成交的价格）**，或者跟踪对手价复杂的回测方法，我这种方法针对日内非超高频可以的。超高频策略主要赚取一瞬间（几微秒）的优势，我这种方法把500毫秒的优势都抹掉了，因此不适合更高频的回测。


#### 3.2 新因子介绍

##### 3.2.1 关于移动最大最小值
移动最大最小，是合成新因子的时候常用的衍生手段。但是本身这个操作对算法的要求会比较高。python的rolling_max方法是直接调用C的，所以速度还可以，如果做过python和R的量化就知道，R往往不适合高频交易，因为速度慢，尤其是计算移动最大最小的时候，R需要自己找C的源码来调用，否则速度极慢。

##### 3.2.2 新因子的计算
第一个因子：标准化收益率
继承因子计算大类，nr(normalized return)因子，标准化收益率，结尾加上.values。注意代码里面的get.signal.pnl函数运行如果出现boolean长度不匹配，那一般是signal的类型有错误。我们所有的signal和factor计算出来都是以ndarray的形式做的储存到pkl里面，而不是什么dataframe。

第二个因子：买卖量差异不平衡
继承因子计算大类，dbook(difference in orderbook)因子，计算使用np.sign()，此处需要注意**精度问题**，这里bid.qty和ask.qty都是整数，所以在python里面，int单纯的加减不会造成精度问题。但是比如在R里面，没有整型，都是float，那么比如相邻挂单量都是5，作差就会出现比如1e-15的情况，那这种时候使用类似取sign()这种方法就会变成-1，产生误差。因此每次要使用到比如sign()这种函数，就需要考虑到一个精度问题，**比如数字货币里面的挂单量不是整数**，那在python也会有误差，使用sign就会有误差，这个时候就需要自己手写一下，比如离零多少范围之内，就认为sign是零这样。之前做SLA的project的时候，有一个地方就是``if total()... < 1e-7: ...``，那应该就是避免的这个精度问题。

因子模版的定义里面使用了修饰器（decorator），在`stats.py`里面。

计算因子步骤就很简单：
1. 定义因子对象：``x3 = foctor_dbook_period()``
2. 创建因子目录：``create_signal_path(x3, product, SAVE_PATH)``
3. 并行计算因子值：
``parLapply(CORE_NUM, file_list, build_composite_signal,signal_list=x3, product=product, HEAD_PATH=HEAD_PATH)``

注意：
- 本次生成因子分布有所改进，使用了移动均线来计算，参考函数`get_all_signal`。也就是说我们采取了**因子的衍生值作为因子值**，先计算好独立的每个period的因子值后，取移动均线moving average去平滑，取平滑后的因子值作为最终使用的因子值。
- 这样计算分布的因子值本质上是过去period个因子值的平均值，比起过去间隔period个抽一个好一些。
- 计算因子采用了一个因子阈值，代码里面有一个类似``open_list = np.arange(0.005, 0.021, 0.0008)``这样，这个是因子阈值，一共100组。**这个因子阈值干什么用的**？
- 回测因子策略，atr_filter统一采用0.01，这个atr_filter是第一章回测策略那个threshold吗？
- 由于事先不知道因子是正相关（趋势）还是负相关（反转），所以reverse=+1/-1要测试一下；**关键看大阈值时的相关性**，这句话什么意思？
- 回测结果保存下来，然后筛选策略调整参数计算就比较快了。
- 以上都是单因子回测，对单因子要求不必太高，**最终是多个因子组合起来看表现**。

其他一些因子：
range.pos：价格的位置；ma.dif.10：双均线；price.osci：摆动指标；kdj.k：KDJ指标中的K；Kdj.j：KDJ指标中的J……
range.pos：价格的位置；类似势能。如果没哟输出结果，说明在训练集上的表现不够好，可以尝试reverse=1。最后结果是KDJ的两个效果都不好，符合预期。**更多的因子可以参考reference书本里面有很多的因子。**

#### 3.3 偏度和峰度
偏度和峰度本身就是因子分布衍生的一种，之前也见过海通的金工报告写过一些高频因子的偏度的有效性检测。正态的峰度是3，我们计算完因子后，会统计因子值的时序上的一些性质，看看自相关性、平稳性等等。那我们也会去看因子的偏度和峰度，**因子的峰度在3、4左右比较合适，因为太大容易过度拟合，交易次数会比较多，取值大的机会更多；而太小说明因子太简单，无法识别交易机会**。为什么？

我们统计之前构造的几个因子在五种黑色系产品的峰度和偏度，发现比如dbook在五种黑色系产品里面的峰度基本都是3-4之间，比较好。range.pos在2-3之间，ma.dif.10在5-10之间，偏大，price.osci.10在3-5之间，可以，hr在3-5之间，可以，kdj.k在1-3之间，可能因子过于简单，太小，不好。kdj.j在1-2之间，更小，不好。

分析了诸如各种统计特征之后，我们的想法一定是如何改进因子。改进因子的思路：
对太简单的因子（峰度小于3），可以通过两个因子相乘使其变得复杂。这时候要求相乘的两个因子一个是方向性因子，另一个不是，这样才有意义，得到的因子是方向性的。两个带方向性的因子相乘意义不大，而对于太复杂的因子（峰值过大），可以考虑**对极端值过滤，就是一些处理outlier的方法，这样可以有效减少峰度，使之变简单。**

#### 3.4 总结
- 由上面的分析，比较复杂的因子，峰值大的如双均线ma.dif.10，可以考虑过滤极端值，或者使用其他周期的均值进行计算，看改进后的统计结果如何。而像峰度小的简单因子比如kdj的两个，需要通过和其他因子相乘的方式，提高复杂度，下一讲会进行这样的因子加工合成。
- 可以尝试其他周期的因子，这就是因子本身的参数优化了，这里一律使用的都是比如4096的周期做的滑动，可以使用更短的周期，也可以做其他的衍生创新，比如遗传规划挖掘等等。
- 因子优化和反复尝试是策略开发中必不可少的环节，各类细节调整、处理，调参，正交，标准化等等，因为因子是机器学习预测模型的基础，如果因子好，其实不需要太复杂的模型；如果因子不好，模型复杂了其实也帮助不大；因此，设计好的因子其实反而是比较重要的，模型都有现成的。
- 另外一个优化思路是尽管因子效果普通，但可以不断迭代选择因子和调整因子的权重，这也是可行的，**这个我就不会了，怎么搞？**
- 看起来使用EWMA对一个取值离散的因子值做平滑是极其常见的操作，因为这样总是能让因子值快速收敛，同时具有很好的时序上的平稳性，较平缓的偏度峰度等指标。比如前面那个取np.sign()的因子就是这样，把-1，0，1的离散序列转成平滑的因子值。


### 4. 批量生成因子

#### 4.1 波动性因子

##### 波动性因子的定义
有一些因子跟方向无关，只与波动幅度、震荡区间宽度有关，这类因子对于期货无法直接给出买卖信号，因为它无法用来指导方向，但可以起到辅助的作用，我们可以称它们为波动性因子。

##### 波动性因子的特征与例子
波动性因子与价格区间有关。某些方向因子可能峰度取值小于3，说明过于平稳，没什么波动，**这类因子有点欠拟合的味道，回测的时候交易次数不多。因此可以乘以波动性因子使之更复杂，峰度可以提高，回测起来符合交易的条件。**

四个常用的波动性因子：
std波动率、range价格范围、trend.index趋势度指数、volume.open.ratio成交持仓比例

快速计算方差：（为什么要写一个快速计算方差的代码？）

```python
def fast_roll_var(x, period):
    x_ma = cum(x, period)/period
    x2 = x * x
    x2_ma = cum(x2, period)/period
    var_x = x2_ma - x_ma * x_ma
    var_x[var_x < 0] = 0
    return var_x

```

计算因子需要考虑的因素：

- 建议使用服务器，内存至少64G，硬盘10T，这样可以全部数据一次性导入内存然后大规模计算因子可以快很多很多。但很多人电脑内存不够大，因此我这里选择每天导入。（什么叫做一次性全部导入？是全部读出来吗？类似用redis内存数据库那样？）
- **未来可以加速的办法：全部数据先导入内存，每次提取一天的数据，计算全部的指标，不必每个指标都重新读入数据，按天进行并行计算**。

意思就是之前我们是一天一天的读取数据，然后计算因子，和我做的leadlag计算HY有点像，就是数据是每天一个，比如每天一个pkl，然后对每个因子，我都要去按天去读取，然后最快的方式也就是对天数并行，相当于每个核干的事情都是读取pkl再计算因子，这样效率很低，因为计算不同的因子的时候数据要反复读取。**然而这个加速的意思就是我先把比如一个产品的所有日线数据全部读取好，读到内存里面，有点像redis一样直接全部读好存到RAM，然后每次提取一天的数据，计算好全部的因子，然后按天数并行，这样就要求内存足够大能把一个产品的所有日线pkl都读好存好，这就需要接近10个G的RAM。而之前那个反复读取的方法，每次读取完计算完因子就自动把这部分内存释放出来了，所以不需要很大的内存**。

#### 4.2 批量生成因子
批量生成因子的方法：
- 预测因子：方向性因子
- 波动率因子：不带方向的因子
- <font color=blue>方向性因子*不带方向的因子=>新的方向因子</font>

注意：
- 两个带方向的因子相乘没意义，原因是如果两个都认为涨，两个都是正号，乘起来正号，没问题；但如果两个都认为跌，两个都是负号，乘起来正号，就刚好预测反了。而且大多数时候我们本就判断不了因子是顺势还是逆势的，这么一乘就乱七八糟了。
- 可以继续扩展，新的方向性因子继续乘以波动因子变成新的因子。
- 新的因子可以加移动均线、上下界等过滤，其实和上讲说的outlier过滤一样，目的是降低因子的复杂性。

程序相关问题：
- 因子相乘之后要加上.values，把dataframe的列转成ndarray。
- 核心函数：``construct_composite_signal(dire_signal, range_signal, period_list, all_dates, product_list, HEAD_PATH)``
- 实际策略开发中，我们研究出了一些因子，必然是优化了这些因子，比如按照上面的各种因子优化方法，标准化，动态调整，正交等等之后，去使用这些基础因子批量合成新的因子，比如上面的因子相乘，还有更复杂的遗传规划，进行更多更多样性的代数组合。这几乎是进阶的必不可少的一步，然而这一步往往极其耗时，在这五个产品的高频CTA里面，我们光使用简单的因子相乘去批量合成，代码都16核并行跑了接近五个小时，更不用说用遗传规划构建因子组合后再取合成计算需要多久，简直不可想象。这一部分因子合成最好用C++，速度最快，和实盘最接近。不然实盘我们现场合成因子，也是需要大量的时间的。


#### 4.3 生成因子分布
前面一直没有搞懂那个因子阈值和分布是个什么东西。保存因子抽样的意义：
我们需要计算开平仓阈值，因此需要先对因子进行抽样。一次性生成了，保存下来，不需要重新生成，节省时间。可以选取全部日子，不需要每10天抽1天，确保结果更可靠。
**什么意思？**

由于每次计算因子都要load data，还要从里面取good=True的部分拿来算因子，太慢了，因此把good=True的列单独保存出来到good_pkl文件夹里面。

#### 4.4 策略回测


#### 4.5 五档盘口因子

**数据介绍**
目前有上期所5档行情2020年5月-2021年5月，时间戳和1档行情可能对应不上，因此独立分析。简单分析dbook、imb因子作为例子。


### 5. 跨品种因子与套利因子
#### 5.1 跨品种因子

**时间对齐与去重**
- 大商所的毫秒是0-999，可以放入一起排序
- 收到一个品种的时间，另外一个品种此时取之前最新的时间对应的数据



**回测跨品种因子交易曲线**
- 策略是用product_x的因子交易product_y

**评估交易结果**
- 步骤跟前面评估单品种差不多
- 结果显示训练集表现较好的策略，测试集未必好
- 大家可以测试其他因子和品种对

#### 5.2 套利因子
**套利因子取值**
- 获取套利因子的分布：`par_get_arb_all_signal`

**套利策略交易过程**
- 一种是各自1手，一种是相等金额
- 无论哪种，开仓之后，手数不再改变，直到平仓
- 注意代码里面手续费计算


**套利因子回测**
- 每次买入和卖出对应的都是组合
- 每次买卖是等资金
- 套利对的选择最好同板块
- 两个品种的数据都要处理，同时交易


未来改进：
- 有可能需要更长的回测周期
- 也可以放宽日内的限制

回测划分训练集和测试集



**套利因子结果评估**
结果不太行。


### 6. 投资组合优化
#### 6.1 第五周答疑
**get.all.signal做了上周的处理之后是不是相当于选取了几个sample的signal，那是不是可以比如说随机选取10个file再看他们的峰度，这样就不用iterate全部的文件**？
- 每10个取1个，每个文件间隔period来取数据，因为很多指标的周期都是period，这里是4096；这样子间隔4096个数据取就可以最大可能的保持独立性；如果只取10个文件，覆盖面不够；比如现在1032个文件，我的方法会取103个，覆盖面更广一些。


**我们基本上用峰度来判断因子的好坏，再比较不同阈值的sharpe，峰度是否和sharpe有直接或者间接的联系？进一步讲，单个信号的正态分布是必须的吗？**
- 峰度跟夏普并没有太直接必然的关系；
- 最简单的道理，可以随机生成任意峰度的指标，它显然跟sharpe是无关的；
- 正态分布更多为了平稳，过去产生的交易未来也会产生；
- 如果一个指标的峰度太高，可能容易过度拟合；
- 因此，可以认为合适的峰度是必要非充分条件。

**我看课上构造因子的过程基本是两个因子相乘，那这样的话和回归模型说complete second order model 中的interaction term差不多。然后我们也可以用stepwise regression的方法来也相当于筛选因子。所以构造复杂因子是否有必要。因为理论上这种构造因子方法可以一直无限构造下去。**

- 一个方向因子 \* 一个波动因子=新的因子，不是两个方向因子相乘，因此跟多项式回归还有点不大一样。
- 当然，这种方法是可以一直无限构造的，就类似遗传规划，python有deap/gplearn这些包可以处理，比如给数据、基本操作符、目标函数，它就可以自动不断迭代来优化。
- 比如wordlquant就有人这么做，可以生成几百万个因子，但后来它们的websim倒闭了，所以这类方法褒贬不一。

**像KDJ这种因子峰度比较低，而且训练集表现好，测试集表现差，应该是过拟合的表现，这种情况我们应该如何处理？**
- 其实lasso属于Sparse model，选择短期有效因子，依赖滚动筛选来盈利。并不寄托于构造一个长期有效的简单因子。
- 不断滚动优化产生的模型本质就是一个长期有效的“因子”，但显然构造过程很复杂，难以人为直接构造
- 未来课程会讲到，**一般lasso这种模型比ridge要好一些**，得益于sparsity


#### 6.2 平均配置
- 根据训练集筛选策略
- 获得在测试集中的表现

结果评价：
- 训练集表现不错，测试集也可以
- 未来用模型汇总因子，并且滚动优化，可能可以好一些
- 



#### 6.3 马科维茨均值方差模型
**基础理论**
- 平均分配资金的好处是计算方便，无论多少个策略都可以；但缺点是没有考虑策略的收益风险比和相关性；但马科维茨均值方差模型可以；

- 不带约束的情况下可以通过公式求得解析解，所有解会在（标准差-目标收益）平面上的一条向左凸的曲线上；曲线的左顶点就是最优解；

- 如果要求不允许卖空，可以加入非负的权重要求，这往往不存在解析解，可以用凸优化技术求解；另外，可以根据实际需要，加入更多的约束条件；


程序实现见`get_weight`函数。其实是《statistical models and methods for financial markets》的第86页的代码。这个是求的理论解。

**缺点**
- 有时候会发现部分品种占比过高，其他品种基本权重为零，因此需要加入一些约束条件。约束条件在`qp.solve_qp()`函数里。加入约束之后无法求解析解，要用优化包。比如python的qp.solve_qp求解quadratic programming。**约束条件：比如最小1e-5，防止做空策略，取零的话会出现1e-15这种数字**。

**总结**
- 优点：考虑了策略的相关性和收益风险比，得到的是样本内收益风险比最高的策略组合

- 缺点：容易过度拟合，权重可能集中于少数品种；

- 改进：加入更多约束条件，限制策略的权重范围

#### 6.4 风险平价模型、夏普比模型
**风险平价的理论**
- 本质上一份风险一份权重，**不考虑收益，甚至可以不考虑相关性**。可以用**协方差矩阵**，问题与马科维茨一样，容易过度拟合。可以只用**对角矩阵**，仅考虑方差，不考虑协方差。传统股票、债券风险平价实质上仅考虑对角矩阵（桥水）。比如只考虑波动，不考虑相关性。股票波动是债券3倍，那么权重就是1：3。见`risk_parity()`函数。

**夏普比分配模型**
- 策略夏普比越高，权重越大，也是忽略相关性

#### 6.5 主成分分析
Principle Component Analysis（PCA）本质上是一种线性降维技术。每个主成分都是所有因子的线性组合。每个主成分是独立的、垂直的。这样因子都是线性无关的。然后抽取方差大的因子组合，删除方差小的。比如按95%的方差选取前面的主成分。

**优点**
避免了过度拟合，因为波动小的主成分删除了。

**缺点**
设置策略权重的时候并没有考虑表现，不一定波动大的表现就好。

**改进**
用主成分产生的因子作为备选因子，在这些因子的基础上用回归模型，这样可以保留因子独立的优点，也可以选取跟因变量相关性高的因子。


#### 6.6 小结
介绍了平均分配、均值方差、风险平价、对角线风险平价、夏普比、主成分等几种投资组合优化的方法。似乎在样本外都不大行，下周加入模型整合因子试试。


### 7. 线性回归

#### 7.1 第六周答疑
**关于本周讲的投资组合构建，是对不同因子根据投资组合优化分配权重，可是实际上还是对一个产品进行买卖，这种策略是否可以用到实盘上？因为有时候可能一个策略买另一个卖，本质上是买了又卖出，反而亏手续费？**
- 同一个品种会抵消，不会下单；不同品种自然是没有影响。理论仓位和实际仓位的匹配，不成交会追单；超高频：信号驱动不成交不会追单。但我们不是超高频，如果这一秒有策略买，下一秒有策略卖，可能会增加费用，但这个已经在回测中包含了。

**我是不是应该先用回归等模型得到交易信号，得到资金曲线，再对其他品种构建类似的因子得到信号，再在不同产品之间进行投资组合的优化，这样的顺序？**
- 之前的对因子进行投资组合不能提高平均单笔利润，也不能提高交易频率；
- 如果先用回归模型整合因子，再做交易，那么是可以提高单笔利润的，也可以提高交易频率
- 本周讲的就是先用回归模型整合因子


#### 7.2 机器学习要素：自变量、因变量、模型
**自变量（因子）**
- 可以是：简单因子、复合因子、波动率过滤

**因变量**
- 可以是：未来一段时间价格变化；对价格变化进行过滤；未来价格变化的夏普比率，风险调整后的收益；不固定时间的因变量，比如说价格涨跌0.1%就划分

比如对数收益率：`ret[t] = log(wpr[t])-log(wpr[t-1])`

代码里面`fcum(data['ret'], period)`函数是最重要的，可以计算未来一段时间的变量累加和，比如计算未来一段时间的收益率。例如计算2999的收益率，是从3000到3000+period，一共period个行情。此处的因变量暂时选用未来4096 ticks的收益率。

`class factor_ret_period(factor_template)`用于计算因变量。
`class factor_ret_period_002/004(factor_template)`是过滤极端值后的因变量。


**模型**
线性回归（本节课讲）
带正则化的线性回归（下节课）
决策树回归（gbm, xgboost, lightgbm）

#### 7.3 因子矩阵
**统计每天的因子数**
`count_daily_num()`

**构造因子矩阵**
`get_sample_signal()`

- 合并各品种的因子矩阵，合并前需要标准化。
- 每个品种的因子矩阵分别标准化，然后再合并，标准化不需要减去均值，不对行情涨跌进行预判，这样每个因子的矩阵才具备可比性。
- 目前缺点：没有用数据库，所有文件保存二进制；跨文件调用可能比较慢，未来或许可以采用MongoDB等数据库优化数据存储。不过安装新的软件又比较麻烦，也怕兼容的问题

#### 7.4 单因子模型

- 单因子简单线性回归

- T值计算：计算单因子t值的程序：`get_t_value()`

- 筛选因子：筛选出t值绝对值大于2的因子

- 因子相关性：计算这些显著因子的相关性
- 前向回归选择因子
- 单因子回测：测试样本建立；对每个品种计算原始系数；构造预测值的分布(all signal)；回测因子表现，reverse统一用1，不必区分正负


#### 7.5 多因子模型
不是说多因子模型就多好。放入一半因子，训练集和测试集都有可能更差。

**这里是多品种的数据一起，可能部分品种会变差，但整体而言，训练集应该会变好。这种最小二乘回归的效果有时候并不理想，容易过度拟合**。

结论：
一半因子的时候，总体上比单因子更好；放入全部因子后，训练集可能好一些，但测试集更差；说明一半因子没有过度拟合，全部因子就过度拟合了。


#### 7.6 不同的因变量的结果
如果直接用Return作为因变量，return的分布高度肥尾，会影响回归的效果，可以通过一定的限制来改进。尝试三种：
- 使用一半因子，限制return在2%以内

- 使用一半因子，限制return在1%以内

- 用AIC选择因子，使用原始的return

结论：
- 1%约束效果很差，首先排除；一半因子2%约束全部品种都有结果，3个样本外盈利，表现不错；；AIC在全部品种有结果，2个样本外盈利。

#### 7.7 独立建模的结果
之前是全部品种的数据合起来建模，还可以考虑每个品种单独建模的效果。此时每个品种分别回归，结果表明rb、i、j还可以，其余一般。


#### 7.8 总结与作业
本周尝试了各种不同的因变量；一开始联合建模，后来单独建模；联合建模的话交易次数可以多一些。


### 8. 正则化线性模型

#### 8.1 Lasso

**最小二乘及其缺点**
Y=a1x1+a2x2+….+anxn，a1,a2…,an没有约束，容易过度拟合，完全最小二乘拟合。虽然是无偏估计中方差最小的，但还是不够小，我们希望方差更小，哪怕估计是有偏的。上周介绍的forward selection，本质上还是最小二乘，系数取值幅度还是会比较大。Prediction error=bias^2+variance，涉及到bias-variance trade-off。

**Lasso的特点**
Lasso:对a1,a2…an的绝对值的和进行约束
Lasso: |a1|+|a2|+…+|an|<=t
使得优化搜索空间变小，拟合程度变差
如果是二维的，搜索空间是一个正方形
如果是三维的，自然就是立方体
n维的，就是n维的多面体
没那么容易过度拟合
得出来的解也不是无偏的
只有最强的因子保留下来，弱的因子删掉了
强的因子系数也被收缩，不会完全拟合
用到稀疏性sparsity概念，小波分析也有类似效果
用CV（cross validation）来选取相应的t值

- 代码加上`fit_intercept=True`，防止样本有偏，分品种进行拟合。
- 先对自变量标准化，解有很多零，说明这是稀疏解，只有最强的因子保留下来，避免过渡拟合。

lasso结果可以，但是品种i的结果很差。

#### 8.2 Ridge
a1,a2..an进行约束
Ridge:a1^2+a2^2+…+an^2<t

Lasso:
变量筛选；
计算复杂一些；并没有直接的公式，需要搜索数值解，目前有人使用admm来求解；
从贝叶斯的角度来看，每个系数是随机变量，先验分布双指数分布（取零概率大），最后删除冗余变量

Ridge:
保留所有变量；
计算简单，直接求逆；
先验分布正态分布，取值为零的概率是零，取零附近的概率大一些，但基本上不会取到零值

a1,a2..an进行约束
Ridge:a1^2+a2^2+…+an^2<t

Lasso可以Sparsity建模，p>>n，5万个因子50样本也可以

股票联动性，因子来源特别多，但样本比较少，需要lasso，适合频繁调整模型，对因子质量要求不会很高

Ridge适合每个因子都比较确定有意义，并且n>>p，不频繁调整模型

a1,a2..an进行约束
Ridge:a1^2+a2^2+…+an^2<t

交叉验证cross validation
Cross validaiton选取lambda，就是类似上面的t，选一个最优的值

如果t取得很大，就等于线性回归；

如果t取得很小，就全部是零；

中间要平衡

用2%过滤的ridge模型，每个品种都有系数


#### 8.3 ElasticNet
Elastic net=lasso+ridge

Alpha=0: lasso

Alpha=1:ridge

0<alpha<1:elastic net, 

l1_ratio(1 for lasso and 0 for ridge) in python 

Lasso:1996
Ridge:更早，非常早，忽略不计
Elastic net: 大约2005
Lasso当年具备一定的革命性质
Group lasso/sparse group lasso/pca lasso/adaptive lasso
Lambda:线性的参数，取值表示模型复杂程度

拟合效果不大好，只有a有系数

**结论**
- 总体效果跟lasso差不多，未来可以直接使用lasso即可，加4%过滤。以后可以考虑用pca lasso，但随着因子增加计算量会很大；Lasso具有因子筛选功能，往往不会有这么大量的因子。


#### 8.4 输出模型到文件
本节课介绍了lasso, ridge, elastic net等模型，加了4%的约束会好一些。Lasso、Elastic net模型整体更好一些。下周介绍整体建模和滚动优化。


### 9. 多品种联合建模

#### 9.1 多品种联合建模
之前是每个品种分别建模，优点是可以更贴近每个品种，缺点是数据量不够泛化能力不行。甚至一些品种用lasso训练得不到有效的模型，系数都是零。今天先研究多品种一起联合建模，多品种一起交易次数更多，不容易过度拟合

合并训练集数据，合并前，各品种分别标准化。运用lasso建模，合并之后再进行一次标准化。运用lasso建模，把系数映射回去原来的品种。

结果表明，联合建模的效果比之前单独建模要好。

#### 9.2 逐月训练模型
**为什么逐月滚动？**
- 对于日内模型，最好可以按日滚动
- 但那样计算量特别大，所以现在先按月滚动，看看结果怎么样

**滚动vs非滚动？**
- 因为实盘过程本质上是滚动的
- 不断有新的数据过来，不可能永远用2018年以前的数据进行训练

训练函数为`get_multiple_lasso_roll_model()`，每个月训练一个新的模型。

#### 9.3 逐月生成预测值
**滚动生成预测值，此时生成的日子仅仅是该模型的训练集和测试集，不是全部数据**！生成预测值的分布，只用训练集的数据计算，每次向前滚动一个月，训练集也每次平移一个月。

**为什么训练集需要平移？**
- 如果不平移，那么开始的数据一直会用到，训练集长度越来越长，可能拟合效果会越来越差；
- 如果平移，可以保持训练集长度不变，拟合效果大体也可以保持

#### 9.4 逐月生成交易结果
每次用不同的模型，不同的阈值，每次测试集也只有1个月。要保存大量的历史数据，比如历史的预测值，占用空间比较大。比如30个模型，每个模型都有约500天的预测数据。

**未来改进**
- 未必用最新的模型，而是滚动对比，只保留当前历史数据表现最好的模型
- 因为模型训练时可能会有很多噪音，很多模型可能并不好，比如相关的数据并不适合生成模型（例如行情太极端），这样保留原来的模型反而是更好的选择

**注意事项**
- 样本内外的时候，只用2019年以前的数据训练，对2020年来说，已经有1年的数据没用到，对于2021年来说，有两年的数据没有用到，所以模型未必适合
- 滚动的时候则一直用到最新的数据，并且删除旧的数据，更贴近当前的行情，所以可能表现更好，也可能训练集不适合训练模型，反而更差，需要仔细研究
- 因此，并不是说更严谨的方法表现更差或更好。实盘的时候，筛选策略时可以更为仔细，有可能比滚动优化表现更好

#### 9.5 总结与作业
可以使用其它模型进行滚动，比如ridge, elastic net。



### 10. 树模型
#### 10.1 gbm建模
GradientBoostingRegressor内嵌于sklearn。1995年发明了adaboost算法；1996年发明了lasso算法；1999，斯坦福的Friedman教授发表了对adaboost的解释性文章，指出它本质上是损失函数为exponential loss的模型，然后提出了gbm的算法。

**GBM的特点**
本质上是线性可加模型，每个因子是一棵二叉树。迭代进行，每次增加一棵树，减少拟合误差，在增加树这方面无法并行计算，只能依次迭代。增加树的时候，系数乘以一个较小的步长，因此具有slow learning性质，避免过度拟合，具有类似lasso的性质。<font color=blue>因此可以理解为每个因子是一棵二叉树的缓慢线性拟合的模型</font>。

**GBM的缺点**
计算上是一阶逼近，收敛速度慢，精度较差。没有使用并行计算，速度较慢。计算结果有时候比较奇怪，正负预测值并不对称，难以用来交易。

**GBM的建模过程**
- 设置cross validation的参数
- n_estimators/ntrees：树的数目（python/R）
- max_depth:树的深度（python)
- learning_rate：学习速率
- min_samples_leaf：最小叶子的样本数

`get_daily_gbm()`计算GBM的每天的预测值。计算预测值的抽样与评估模型。


#### 10.2 xgboost建模
**xgboost介绍**
华人研究出来的计算模型，大概2015年，更快速计算gradient boosting machine，kaggle上目前最热门。

**xgboost的特点**
- 并行化：计算每棵树的时候并行，提高了速度，但依然是一棵一棵树迭代计算
- 二阶逼近：gbm是一阶，xgboost是二阶
- 还有一些小技巧, subsample，避免过度拟合
- 几十个参数，很多不需要调，自动设好较优值
- 一般需要调整的：<font color=blue>树的深度，树的棵数</font>，（学习速度），(最小的叶子数目)等，跟gbm一样。

**xgboost和gbm的比较**
- gbm由于收敛速度慢，预测值偏小，欠拟合明显，交易次数较少
- xgboost收敛速度快，预测值偏大，有过拟合现象，交易次数较多
- 此次建模xgboost表现还行，gbm滚动效果不好，gbm样本内外表现较好

#### 10.3 lightgbm简介
微软开发的gbm模型。很好用很火，万金油。

**lightgbm vs xgboost vs gbm**
- 一般认为lightgbm速度会比xgboost快
- 但凡是这些模型对比的文章都有一定的倾向性
- 比如它要证明xgboost比gbm好，可能花在xgboost的调参时间比gbm多的多

**lightgbm vs xgboost**
- 也可能作者自己更擅长lightgbm，最后大家可以自己用熟一个就行


#### 10.4 总结与作业
介绍了gbm/xgboost/lightgbm模型，xgboost表现不错，调参也需要经验。作业尝试用lightgbm在建模，仿照前面的步骤。


### 11. 集成建模
#### 11.1 ensemble建模
前的模型只考虑了除以period余0的点，这样虽然有独立性，但是大量的点没有用到。如果把除以period余1、2、3、…、period-1的点都分别独立建模，则模型数目又太多，怎么办？

**ensemble建模方法**
除以period余0的点抽出来拟合系数（之前做法）
除以period余400的点抽出来拟合系数
除以period余800的点抽出来拟合系数
……
除以period余3600的点抽出来拟合系数
然后把拟合的系数取平均，作为集成模型（ensembled model)


**优点**
- 是每个模型拟合的时候样本都是独立的，系数计算不会出问题
- 最终也只有一个系数，跟之前的模型一样的，不会影响实盘运行

**缺点**
- 只适用于普通因子，树型模型可能不行，因为每棵树的结构不一样，不能直接相加
- 如果不同模型训练的树全部加起来，那么因子数目会异常庞大，降低实盘运行速度

计算多个因子矩阵的函数：`get_multiple_sample_signal()`；把因子保存在3维array里面，有一维是除period的余数；求解10个模型。之后过程差不多，计算预测值、分布、回测等。

**总结**
模型的稀疏性会丧失，基本上每个因子都会取到；所以不同的数据之间还是存在一定差异的；最后结果还可以。

#### 11.2 另外一种ensemble
之前是先取系数平均值，然后得到一个模型，也可以每个模型作出资金曲线，然后再取平均值。这样的好处可以增加容量，因为每个策略下单的时间点不大一样。
步骤：生成模型、回测模型、生成交易曲线

**总结**
主要都是避免单用除以period余0的样本不大好；用多几组样本可以降低方差两种都可以，效果差不多。

#### 11.3 中低频隔夜交易
- 按合约进行划分，合约结束一定平仓
- 日内是按天划分因为收盘一定平仓
- 中低频一般加入平仓阈值
- 缺乏日内强行平仓的机制，有可能持仓很长时间都不平仓

**步骤**
统计每个合约的天数：`get_contract()`，生成训练集和测试集、拟合模型、查看每个品种的系数、生成预测值；按合约回测，换月才平仓，否则仓位不变；计算隔夜持仓的pnl，加上隔夜持仓的pnl，回测。



#### 11.4 总结
集成模型、中低频交易

作业
可以尝试滚动优化

目前为止，所有的知识结束了，下一周的课程是完全样本外的数据来测试，2021年5月-2021年8月数据。


### 12. 样本外统一测试

#### 12.1 整理新数据
测试步骤：整理新数据、计算因子、计算因变量

2021年5月-2021年8月的新数据，一共1126天。之前遗漏5月10日-5月14日的数据，目前补充上了，所以可能有点混乱。不少因子可能需要重新计算，特别是5月7日和5月14日，如果有问题大多是这两天，大家可以重新生成这两天的因子。


#### 12.2 样本内外测试与滚动优化
使用两个模型：
模型1：第9周的样本内外lasso，2019年之前数据训练。只针对新数据计算预测值，回测策略

模型2：第9周的滚动lasso，按月滚动优化，计算预测值的分布，滚动回测策略

#### 12.3 未来规划
大家可以联系微信babyquant报名C++课程
可以这边开户，top 5/20(AA/A)级期货公司，可以网上开户
穿透监管在C++课程会讲，统一开户也好处理
我使用ubuntu 16.04 lts，上海机房腾讯云
一个月60元左右，不是给我，给腾讯云
我可以通过镜像来传递程序，linux系统比较复杂
大家可以把ID给我就行，ID是纯数字，我共享镜像，大家重装系统
Linux的编辑器有emacs和vs code，vim是自带的
有windows和linux两种选择
Windows用visual studio 2017

见两个PDF。

