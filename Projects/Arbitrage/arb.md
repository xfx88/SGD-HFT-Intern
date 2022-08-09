<font face=''>

<div align=center>

## Statistical Arbitrage
</div>

---

<div align=center>

### 统计套利策略研究

</div>

### 1. 经典文献阅读

#### 1.1 前言
这个章节跳出leadlag的框架，回归一下统计套利的大方向、本质理解以及各种不同的玩法，从基础的pair trading晋升到更加复杂的深度学习对价差时序建模等，彻底了解套利的主流玩法，便于对leadlag策略产生一些启发，同时想办法implement到币圈去，因为币圈我之后实盘的主要策略方向还会是套利，无论说期现套利，或者各种**中性策略**本身也属于对冲套利，都会是未来一年实盘的重要方向。

同时借这个机会把ML、DL的工具用起来，gpu的pytorch已经下好，虽然CUDNN还没有安装，一个是因为Ubuntu的20.04的版本在NVIDIA里面只找到了对CUDA的11.4有安装，但是linux服务器预装好的CUDA是10.2版本的……而且下载需要使用sudo命令，我没有被root添加使用权限，所以就没有安装。但现在是可以用显卡训练模型了，没有CUDNN也可以用GPU，但是CUDNN的主要作用就是加速训练，不安装的话可能速度会跟不上。这个问题之后再说，比如给我添加一个sudo权限然后下载一下CUDNN。总之尽快跑起来。

#### 1.2 对统计套利的理解
<b> <font color=blue> 统计套利是一种基于相似私产阶段性的价格偏离获取收益的策略，通常分为资产配对、价差时序建模、组合构建三个部分。</font> </b> 套利和对冲/中性策略始终是息息相关的。中性就是利用负相关性对冲风险，从而目标是收获尽可能无风险的收益，本质目标就还是一个无风险的套利。

- 最基础的套利方式是配对交易，考虑各种pairs，做趋势，二者出现价差，然后价差在一定的轨道范围之外，比如假设正态然后在两个标准差外就多空操作。这都是默认二者同频变化，即使不同频也默认信息差会及时得到弥补。
领先滞后关系leadlag也是一种统计套利。当我们去关心一对pairs中谁更先引领行情的发展时，我们会发现A,B中总是A先动，B后动，这种情况下，我们也许不必要交易价差，而是通过A的信息来预测B，交易B。这个时候本来研究的pairs就变成了做单边交易，A和B的pair trading就变成了B的趋势预测，然后交易B。
**统计套利的本质：统计指的是使用的计算fair price的统计模型，套利是使用现价和fair price的差距进行套利**。价格变化当然是随机的事情，模型也有失效的风险，但都是随机的，我们只要相信自己的模型，去测试和backtest就行了。
但是leadlag本质同样也属于统计套利，因为根据定义，leadlag也是在利用pair之间的信息，通过统计模型（就是leader的信息建模），计算lagger的fair price，然后利用差值进行套利。
但是**为什么说套利的本质是单边做趋势**？需要好好想想。
- 币圈里面，非常经典的套利玩法就是跨交易所套利、跨期套利和期现套利。合约和现货整体走势一致，但是独立盘口，独立盘口就会存在价差，也就是基差。那这个时候都在拿什么股指期货做对冲，那为什么不去研究币对和他们的合约之间的lead lag呢？我觉得在币圈这种lead-lag应该是比较显著的。衍生品种类多，而且玩法多，很容易存在lead-lag，关键就是怎么去制定交易策略了。
- 其他的名词，比如跨品种套利一般说的是不同商品期货品种之间的套利，而跨市场套利那就是不同交易所的套利。
- 当然，我们接触最多的套利，可能还是在构建若干portfolio，然后利用portfolio之间构建**多空组合**，然后检验超额收益的显著性，那其实这个过程就是一个套利的实验，因为多空组合就相当于我们初始没有钱，但是通过多空有钱了。我们之前做单因子检验的时候，最简单的就是截面划分portfolio，然后持有看收益画曲线，而更严谨的就是像IVOL因子检验那样，我们通过对fama三因子回归取残差计算IVOL，然后排序，然后根据不同股票对IVOL的因子暴露值不同，取设计权重构建多空组合，检验超额收益的显著性，那这本质就是套利，因为根据套利的定义，我们认为IVOL小的组合收益高，IVOL大的组合收益低，那么意味着当前IVOL小的组合的价格是被低估的，也就是低于其fair price，IVOL大的组合的价格是被高估的，也就是高于其fair price，所以我们就做多IVOL小的，做空IVOL大的赚取收益（稍微有点牵强，不过突然发现IVOL这个计算的回归过程就很像一个和常见风格因子正交的过程，剔除了市值、账面市值比等因子的贡献部分）。当然美股可以多空，A股T0也可以多空。<b><font color=blue>总之我认为截面构造多空组合属于套利范畴，之前是pair，然后策略是一个做多一个做空，现在就是放大到两个组合，一个做多一个做空。而且抛开什么统计模型计算fair price，套利最本质的内涵就是空手套白狼，就是无论市场是下跌还是上涨，我都能赚钱，而且都是赚取的无风险/低风险收益，就是从一个0的初始值能产生一个正利润，FIN2020也讲过，只要当前的初始资产为0，然而未来存在一种possible state下的收益为正，那就是套利。这就是套利的本质，就是neutral，就是中性，就是对冲，多空组合这种属于money neutral，对冲策略则是risk neutral，套利就是中性就是对冲。</font></b>




#### 1.3 Paper: Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500

文章写的很好，策略讲的极其清晰明确，所有模型，数据的调参细节都非常清楚。2016年的文章，引用量几乎是套利里面最高的，有六七百的样子。主要因为简单易懂。

**数据和特征工程**
- S&P500日线数据，750交易日数据训练，250交易日数据测试。没用滑动，只使用了不同周期的lag return作为feature，一共31个feature，lag period从1到20，然后40开始间隔20到240，最大跨度对应一个一年的lag return。数据选取时间段是1990到2015，由于在滚动的用三年的训练，一年的交易，相当于从1993年到2015年，每年都在用前三年的数据训练好的模型，去根据现在的日线数据去预测然后日度调仓交易。即使是每年都在用相同的训练模型，**最后结果是一个ensemble的模型的废后年化有七十多，不过这个也不足为奇，因为文章结尾也说了，主要的收益都是在2008年以前赚的，在2008年到达的顶峰**。这个正好和文艺复兴基金那段时间赚的七八十的年化是匹配类似的！也就是机器学习在08年前特别赚钱，后来收益就越来越低了，因为都在用ML在做预测了，不然也不会至于这种一整年都在用同一个模型还能赚这么多钱的……
- 这里的feature可以理解为简单的动量特征，因为lag return本身就是一个动量的反应，不同period对应不同时期的动量。如果是日度数据，那么这些动量特征其实是很经典的，就是先取daily lagged return从一天取到20天，然后是monthly return，从1月一直到12月。
- 注意预测的target，target不是预测涨跌或者预测收益率，而是预测是否超过了S&P的平均收益，二分类问题。不使用回归问题，因为文献普遍表明回归的效果远不如分类的效果。而且这样的预测标签并不影响策略的制定和收益，因为策略里股票的选取是根据predict的概率来的，而不是这个0，1的标签，此外策略是多空的，也就是不会说熊市都在跌那就没钱赚了，这种多空本质上就是套利，毕竟套利最本质就是空手套白狼，这种dollar neutral，初始资产为0最后多空换来了正收益本来就是套利，或者就叫美元中性策略。




**优化器时间线**
- 1951年：SGD。`A stochastic approximation method.`
- 1986年：Momentum。`Learning representations by back-propagating errors`
- 2011年：AdaGrad。`Adaptive subgradient methods for online learning and stochastic optimization`
- 2012年：AdaDelta。`ADADELTA: an adaptive learning rate method`
- 2014年：Adam。`Adam: A method for stochastic optimization`
- 2016年：优化方法综述。`An overview of gradient descent optimization algorithms`
adagrad的最大缺点就是不能从local minimum中挣脱出来，而SGD根据其随机性，是可以从local min中挣脱出来。

**DNN**
- 使用了dropout和adadelta方法。AdaDelta优化器是在adgrad上的改进，目的是改进adagrad过激的单调下降的learning rate。 Adadelta不积累所有过去的平方梯度，而是将积累的过去梯度的窗口限制在一定的大小。同时adagrad对超参数的选取非常敏感，尤其对global learning rate非常敏感，而adadelta做了robust处理，对超参数并不敏感。dropout是为了防止过拟合。
<div align=center>
<img src="./figures/adadelta.png" width=70% />
</div>
- 结构是31-31-10-5-2，input dimension对应feature数量，第二个31说是根据heuristic，使用和input一样的dimention可以避免过拟合，第三第四就强制降维。hidden层的dropout取0.5，input层的dropout取0.1，就是有一定概率扔掉一些feature。DNN训练加了L1正则化，rate为0.00001，也就是对cost function加了个L1正则化，目的也是防止overfitting。L1就是lasso，参数会直接逼近0。训练400个epoch，同时使用earlystopping机制，只要最近5轮的loss的平均值，连续5个scoring event下没有提高，就停止训练。作用也是防止过拟合。

**Boosting**
- 用的是adaboost，每次用一个weak classifier，就是一个小的生成树。在stochastic gradient boosting的基础上做变化，使用类似random forest的方法，每次split的时候选取一部分子集的feature。一共四个参数：1.boosting iterations，2.tree depth，3.learning rate，4.number of subset features used at each split。不知道怎么实现这个奇怪的随机森林+boosting的东西。iterations取100次，防止overfit；每一轮的tree depth是3，不取1是为了有一个two-way interaction；学习率公式似乎和神经网络类似，和iterations是反比，常数项是10左右，所以取10 / 100 = 0.1；每次用15个feature，也就是一半去训练。
- 是我肤浅了，看sklearn的document里面，`sklearn.ensemble.AdaBoostClassifier()`和`sklearn.ensemble.GradientBoostingRegressor`以及`sklearn.tree.DecisionTreeClassifier`，里面的参数说的很清楚，adaboost本身的参数有iteration的数目，就是`n_estimators`，有adaboost的算法选取，有{‘SAMME’, ‘SAMME.R’}两种，有`random_state`和`learning_rate`，同时需要传入一个`base_estimator`，这个参数就可以自己指明每轮训练使用的weak classifier，如果是None则默认是一个树桩，否则可以传入比如DecisionTreeClassifier，那么对应的参数就有`max_depth`就是上面说的，各种其他的feature比如说`min_sample_leaf`，还有`max_features`，这个max_features对应的就是The number of features to consider when looking for the best split，也就是上面的那个每次选取的子集feature的数目！！也就是sklearn本身就可以实现这些东西。而如果是GBDT那参数就更多了，有`learning_rate`，有`n_estimators`，有一个参数叫做`subsample`，这个是adaboost没有的，而这个参数如果小于1那么就相当于实现了stochastic gradient boosting，也就是下一次只选用一部分的subsample去拟合，而adaboost没有这个模型因为adaboost定义是需要每次都用到全部的样本。同样也有`max_depth`默认为3，实现每个独立的weak classifer的定义，同样也有`max_features`就是对应每次split取多少个feature，可以自动选择"auto"，可以自己指明整数等等。也就是以上的各种功能全部都能很好的实现，还有很多其他的参数可以调。
- 总而言之就是注意boosting本来就是一种sequential性质的模型，所以每次用到的base model一定要简单，不能复杂。

**Random Forest**
- 常规经验参数，训练1000个树，不会overfit，每次树的max_depth都是20，然后每棵树都选用根号p个feature，也就是根号31。

**Ensemble**
- 使用了三种集成方法，这个集成还和之前学的投票分类器不太一样。sklearn中的`sklearn.ensemble.VotingClassifier`，决定软硬的参数是`voting={'hard', 'soft'}`，硬投票分类器是对每一个prediction，都取majority vote，比如两个投1，一个投0，最后的分类结果就是1；软投票分类器就是根据概率，比较科学，比如两个投1，一个投0，但是投0的预测概率是0.999，那最后分类结果仍然是0。注意只有votingclassifier有这个性质，votingregressor没有，只会把各个分类器的结果取平均输出，最多有一个参数`weight`，决定不同的子分类器的权重。
- 这里的集成ensemble是作者自己写的，把RF，Adaboost，DNN的预测的probability取出来，自己加的权重得到新的probability，然后看是否大于0.5，大了标签就是1，小了就是0。这个程序也很好实现，知道三个分类器的预测概率即可。然后不同的权重可以得到不同的ensemble模型，而每一个不同的ensemble模型就是一个单独的模型。这种概率加权的方式和投票分类器显然不一样，而且听起来还挺科学的，未来我也可以试试。提了三种ensemble的权重的方法，1.等权重；2.根据模型在整个训练集上的基尼系数加权，基尼系数越大说明模型区分度能力越强（见文章批注）3.根据基尼系数的ranking，取到数做加权。
- <font color=blue>使用ensemble的意义:一个是均摊风险，防止某个模型不适用而影响整体，另一个更科学的原因是说，这些单个模型容易困在Local minima，比如神经网络，优化器选不好那就会困在local min，比如adagrad，甚至是adadelta也很容易困在局部最小值，甚至是SGD，也有这个风险，更不用说树模型，无论是adaboost还是随机森林，都使用的是贪心算法，那贪心就是最容易困在一个local min。所以很多机器学习竞赛最后各种模型都试过之后都会加一个手动的集成，比如之前optiver那个lgbm-baseline，最后就输出的是一个KNN和一个神经网络预测值的平均，那个就是一个ensemble，当然和votingregressor也挺像的。</font>

**交易策略**
策略就是一共3+3=6个模型，对测试集每一天，计算之前的需要的lag return，然后扔进已经fit好的模型里面给出预测，六个分类器，每个在每天都能给出一个预测的概率，我们根据这个预测的概率排序，然后做多前K个股票，做空后K个股票。K是一个参数，当然自我感觉也可以考虑根据预测的概率然后加一个阈值来做多做空，或者划分组合，做空top的做多bottom的，这个就比较灵活了。然后比较这六个模型哪个好，顺便比较下K值对结果的影响。每日预测，每日调仓。



**模型结果**
- 六种模型无论在任何K值（取了10，50，100，150，200），directional accuracy都大于50%，注意文章预测的target variable是是否股票表现超过平均水平，而不是是否涨跌，所以accuracy是和实际是否超过平均水平计算的一个准确率。说明平均而言模型的预测准确率高于random guess，一般都是五十二五十三左右。K越大，average daily return越低，毕竟我们引入了更大对于涨跌的uncertainty的股票。
- ensemble模型普遍比单个模型表现要好。集成模型表现优于单个模型的原因是：1.子分类器足够的diversify。也就是不同模型的error的correlation应该低相关。DNN，RF，Adaboost本就是完全不同的模型，这点可以得以保证。2.子分类器全都足够准确。这里三个base learner都达到了不错的超过50%的预测accuracy，那合在一起也不会差。三个不同权重的ensemble表现类似，其中simple average就表现很好。
- 树模型相对比较好训练，随机森林的表现优于梯度提升，又优于DNN。猜测DNN的参数太难调，feature中存在太多的noise，但至少文章提出的这个baseline model表现并不好。
- 模型之间的对比就用一个panel计算了K的影响，不同K的平均daily return，日收益的标准差，accuracy，然后固定K=10下详细统计**几种模型在费前和费后的多空收益、总收益、总收益std、t值、收益分位值、偏度、峰度、PT检验、NW检验，不同分位数的VaR和CVaR，最大回撤、calmar ratio，夏普比率，sortino比率。**
- 几种模型表现的特点就是，VaR比较大，比pair trading大很多，当然收益也大。另外就是最大回撤特别大，基本都在70%以上，甚至DNN的最大回撤达到了95%（就TM离谱这个回撤，基本上就是全部亏完了，跟ETH一样呢。不过基本上见过的ML的策略都是回撤大的离谱……），然后算下来calmar，也就是收益回撤比有个0.99，就是收益大回撤大，显然不是一个好的策略。然后夏普有个1.81，全靠return撑起来的。**这个收益回撤比的意思就是，发生一次这最大回撤，我得用整整1年的时间才能赚回来，才能recover，因为calmar的分子是年化收益**。这个回撤太劝退了，虽然年化收益ensemble也有个七八十。
- **文章进一步计算策略的alpha值，也就是对systematic risk的宏观因子做一个中性化，检验残差的显著性**。把等权ensemble的after transaction cost的收益和传统的fama三因子、五因子，还有自己加上的短期momentum和reverse因子做多因子回归，还有加上了VIX相关的一个dummy variable回归，看模型收益在这些宏观因子上的暴露情况。文章里面贴了fama的三因子五因子下载的地方，fama的website。**回归后的残差就是策略的收益中不能被系统性风险解释的部分，就是我们说的alpha**。回归结果发现，常数项alpha都是显著的，同时其他一些宏观因子的loading也是显著的，对与市场回报率因子的loading是显著的，因为我们的策略是dollar neutral的，不是market neutral（也就是portfolio的初始资产为0，但并不是组合相对market的收益为0）。**同时自己添加的短期动量和反转因子是显著的，这个很好理解，因为本身机器学习就是捕捉趋势的，但是复杂的ML模型同样捕捉到了动量和反转以外趋势信息，所以去除了动量和反转因子后仍然有显著的alpha，不过alpha比用其他因子模型就要小了，因为一部分被动量和反转解释了**。VIX也有一定的解释力，**VIX本就是专门针对标普500制定的隐含波动率指数，volatility inflation，体现对未来30天的市场波动率的一个预期，也叫做恐慌指数（investor fear gauge）**。这个波动率指数系数也是显著的正数，说明ML模型成功捕捉了一些波动率上的信息，波动率越大，ML模型表现越好。

**分时段分析**
- 笑死了，这个彻底蚌埠住了。我就说这些简单的模型，一个二分类怎么会有效。原来1993-2001年赚取了几万倍的收益，然后后来越来越不行，2010-2016年直接净值从1跌到了0.3，而这段时间市场净值都翻了一倍，从1涨到了2，真TM丢脸，真就亏成狗。简言之就是本文的策略没有任何的卵用，上面所有好看的数据全都是靠二十年前回测堆出来的，一大堆什么fancy的ensemble，好看的收益全部都是二十年前的，垃圾东西，我就想这个东西怎么会有用呢？况且九几年赚的那么疯狂，他最大回撤还能到百分之九十多，收益回撤比小于1，所以说是真TM没用啊。笑死了，只能说从文章里学到了很多关于机器学习模型的话术，分析方法，怎么把一个套利策略给设计的能写这么长一篇文章，顺便帮忙温习了一下简单的ML，为下一步搞模型打基础了吧。笑死，除此以外没有什么作用了。
  
**总结**
- 还是学到了很多话术和机器学习简单模型的复习的。文献里出现这么简单的ML模型的真的不多了。重点温习了机器学习，别的策略上给了点insight吧，怎么ensemble模型一类的。
- 最后结果奇差无比，回撤也大的吓人，不过可以理解，因为一方面ML模型基本上回撤都不太行，另一方面本文的模型实在太简单了。三点，1.拿三年的回测，然后一整年都不换模型，这效果能好？注定会死的很惨啊。2.feature都太过简单，一共31个feature，竟然全部用的lag return，任何其他的价量信息都没用，甚至没有用任何的统计函数去构造一些因子，只用了一个收盘价序列，效果怎么好？而且因子也完全没有处理，就直接拿来用了，没有任何的滑动平均优化，没有因子挖掘等等； 3.模型也都是最简单的模型，基本这种单纯的神经网络表现奇差也能理解，因为以前的文献就没见过DNN和LSTM在预测股市上有什么作用的，果然表明结果差的一批，收益也低，回撤还九十多，笑死了。哪怕是加了什么earlystopping什么dropout，什么ADADELTA算法，没个卵用。其他的树模型也就仅仅是简简单单的设置了一些参数，大部分都是default，也没有做任何样本内外的参数优化，什么网格优化，K折交叉检验超参数调优，贝叶斯优化，随机优化什么都没做，全部都用的default，模型这么简单也不会好。至于这个训练label是不是不行我不清楚，感觉还好，这种简单的多空策略，毕竟用到了预测的概率去制定的。


**补充一些知识**
- VaR在险价值，CVaR条件在险价值。
- 基本上各类机器学习模型都可以加一个regularization。比如神经网络，文章里的DNN就加上了一个L2的正则化，这个在torch或者tf里面应该很好实现。同时logistic也可以正则化，在sklearn的`sklearn.linear_model.LogisticRegression`中，参数`penalty`的默认值就是l2，也就是l2正则化。所以之前做作业看到sklearn的结果和statsmodels的logistics不一致，原因就在于sklearn自动加上了L2的正则化。


#### 1.4 Paper: Statistical Arbitrage in Cryptocurrency Markets
2019年的文章，基本是在仿照上一篇。引用量有个三四十。除了使用的是分钟数据，然后把数据周期换了一下，特征基本一致，每次预测是未来两个小时40个币种中超过average表现的情况，然后取前三名后三名做一个多空组合，用的模型是logistics和随机森林。logistics用的L2正则，优化器使用的是L-BFGS。

文章有代码，代码在[Statistical-arbitrage-in-cryptocurrency-markets](https://github.com/Exceluser/Statistical-arbitrage-in-cryptocurrency-markets)建议好好读读，代码写的很好，除了没有数据，基本完整的逻辑都实现了，预测，包括数据处理和模型都有，我提了一个issue给作者要数据，如果不给也没办法，可以试着用ccxt接到数据复现一下，但整体代码写的很好。


#### 1.5 Paper: Deep Learning Statistical Arbitrage

**文章介绍**
- 2021年最新的文章，没什么引用量，但是文章的模型看起来很fancy。之前量化投资机器学习的公众号发过有NVIDIA的人解读这篇文章，说是提供统计套利的最新方法和动向，当时没参加。没什么引用量但是github上的star数量倒还有好几十，看来还是挺多人感兴趣的。
- 文章有代码，作者公布代码在[https://github.com/gregzanotti/dlsa-public](https://github.com/gregzanotti/dlsa-public)，可以通过github desktop直接拷贝到本地了，不过模型看起来比较复杂，并且repo里面没有给数据，issue里面去年都有人提了，估计是没戏了。而且里面大量的TODO的注释，说明代码并没有完全优化。不过本身代码就用了很多fancy的东西，我还不是很了解，比如yaml的模型config文件，还有各种数据库储存等等，有空还是学习一下吧，感觉写的class非常正统，非常符合这种paper的源代码格式，各种文件，class的封装，写的非常规范，包括写论文包括实习的文件都应该这么布置。   
- youtube有作者presentation slides和presentation现场，结合pre去学paper效果最好，而且如果未来要做比如整理我这段实习所了解到的全部有关套利的东西，那么这个pre slides还能用的上哈哈。 


**前言**
- 统计套利的simplest form是pair trading，找到similar asset，利用cointegration关系，对价差建模，超过一定threshold就做空winner做多loser，这样在价差产生mean reversion的时候就可以平仓获利。**本文主要回答两个问题：一个成功的套利策略需要什么成分；以及当下的市场还有多少realistic的套利机会**。
- cointegration，协整关系。协整可以理解为平稳性的进阶版，在课内我们只讲到了平稳性没有讲协整关系。我们在之前做CTA时序因子的时候都在检验因子的平稳性，包括会检验期货时序对数收益率的平稳性，**因为只有收益序列是平稳的，才有预测的可能，才有回归分析的意义，否则所有使用收益作为Y值的回归都是伪回归，因为过去发生的事情未来根本不可能发生**；而因子是平稳的就有好的性质，过去发生的事情才可能继续发生。平稳序列就具有mean reversion的性质，而pair trading常说的两样资产的价差平稳属于一阶的简单系数的协整，协整更广义的是只要两个序列的线性组合是平稳的就可以。检验协整关系使用coint，检验平稳性用单位根检验，最典型的单位根减方法就是ADF检验。使用statsmodels检验协整关系：
```
# coint函数检验协整关系，p值越小代表协整关系越强，H0是没有协整关系
from statsmodels.tsa.stattools import coint
p_value = coint(X, Y)[1]
```
**统计套利的三要素**
1. Arbitrage portfolios。套利最本质的就是空手套白狼，从0到正收益。那基本会涉及多空（leadlag除外）。配对交易的多空都是单个资产，但更广义的多空资产应该是两个组合。如何选择多空组合，也就是如何定义similar？（简单的配对交易就只看时序的相关性，比如日线同频的话只算一个correlation，高频不同频的话用previous tick或者HY去衡量）
2. Arbitrage signal。我们要对价差，或者一个线性配比下的价差做时序建模。那么用什么复杂的模型建模？以及建模后我们要确定交易信号，什么时候开仓什么时候平仓？（简单的配对交易就是价差是一个white noise sequence，那么服从近似正态分布，在超过一定的置信区间范围就会开仓和平仓）
3. Arbtrage allocation。把market frictions, commission, microstructure noise, market impact等各种东西当作constraints，那么我们要做的应该是一个优化问题，就是在给定的这些constraint下，如何进行仓位管理，如何设计交易模式和交易量去maximize a target function（trading objective）？

**当前的困难**
1. **Large number** of assets with **unknown similarities**
2. **Complex time-series patterns** in price deviations
3. Optimal trading rules are complicated and **depend on trading objective**

**一些想法**
- 本身similarity就很难定义，单纯就return的时间序列上，可以定义说线性相关，或者什么时序上的相关性比如傅里叶、互相关，HY相关，但我们一般pair trading也会找同一个板块的股票，那就是股票本身的基本面，流动性等等一些characteristic。这就比较难以定义，但也是在衡量similarity的时候需要考虑的因素。
- 我们之前美赛就是做的对交易策略建模。往往这个过程中就发现很难有数学的部分，包括读了这些套利的文章，基本上数学的部分集中在比如HY函数的底层逻辑（结合到价格模拟的GBM的一些伊藤积分，还有一些线性回归模型的矩阵运算和推导），其他文章基本上数学的部分只会在机器学习模型的时候结合着模型写一些公式，大多数情况数学都很少，这也是我们当时建模遇到的困难，因为平常的因子选股，策略开发，简单的套利策略都不是很好量化成数学公式，都是一些用来理解的交易逻辑，包括CTA，基本上时序因子挖掘，最多就因子计算上面设计到一些数学公式，EWMA等等，还有因子挖掘会有一些公式，模型上有一些公式，真正的逻辑，交易逻辑，回测这些都没什么公式可以写。大量的数学公式集中在衍生品的pricing model里面，这些期权期货的定价模型需要数学理论支撑，但是交易策略本身没有。除了这些基本上涉及理论最多的部分就是组合优化，组合优化里面涉及到的马科维茨均值方差、风险平价、对角线风险平价、夏普比调仓、主成分分析、什么凯利公式这种。此外就是一些评价指标计算，比如VaR，sortino比率，calmar等。当然机器学习模型本身那数学挺多的，讲优化器原理什么的。当时美赛的数学部分全靠1.因子施密特正交化和因子挖掘；2.马科维茨优化。用到的也基本就是投资组合的理论，我们FIN2020和3080讲了马科维茨，以及对应的一个QP问题，他们4800讲了一个针对risk preference的优化问题，把整个做成一个优化问题，当然这个优化问题本身没有考虑手续费，虽然是一个constrained optimization，而只是按策略交易之后再去扣除手续费看结果。
- 但本文的数学公式很多，很适合拿来做建模（MLDL的模型在实盘就可能比较困难，尤其是要处理组合优化后的小数点权重的问题等等），涉及到投资组合理论（多因子模型），涉及到简单的深度学习工具，以及投资组合权重优化。
- 在之前的两篇文章当中，使用的ML模型只是在预测是否outperform the market average，然后去构造多空。但是如果是对价差建模，那就不是一个简单的prediction或者classification问题。本文中**We use a trading objective on residuals of asset pricing models**。也就是相当于是最终还是在解决一个优化问题，和之前的美赛东明做的组合优化问题类似，比如objective是一个risk-adjusted return，像夏普，或者其他的函数，当时美赛的objective选取的是一个mean-variance objective，本质就是自己搭建的一个utility function。然后再mean-variance下这个utility函数取的就是课内的最经典的（FIN2020和STAT4012都讲了的），去在给定的return下最小化Variance。权重大于零就是不能short sell。之前也是因为这样一个明确的优化问题，东明才会去想着用强化学习的模型去拟合一下。
$$
\min_w w^T\Sigma w, \text{ such that } \mu^T w = z, 1^Tw = 1,w \geq 0
$$

写一下KKT条件，等价于（算是duality吗？）
$$
\max_w \mu^Tw-\frac{1}{2}\gamma w^T\Sigma w, \text{ such that } 1^T = 1, w \geq 0
$$
本文用的模型与这基本大同小异。

**方法综述：Deep learning statistical arbitrage**
1. Statistical factor model including characteristics to get arbitrage portfolios
2. CNN + Transformer to extract arbitrage signal: Flexible data driven time-series filter to learn complex time-series patterns
3. Neural network to map signals into trading rules: generalization of conventional "optimal stopping rules" for investment
 Optimize and integrate for global economic objective: **Maximize risk-adjusted return under constraints**
 Most advanced AI for NLP for time-series pattern detection
 
- 套利的对象选取的不是简单的一个预测的多空，而是通过和风险因子模型（加上自己定义的conditional/unconditional的统计因子比如short-term reversal/momentum）计算取的残差组合。
- 时序模型建模并未使用经典时序模型比如GARCH或者ARIMA等等，因为这些模型往往都需要提前在一段时序上fit好，也就是提前把parametric models的参数固定，再去fit，而且就算我们用什么滑动窗口动态调arima，基本上时序的参数也不变。也没有用一些比如GAM, general additive model或者其他的一些regressor去fit，因为作者觉得这仍然相当于是在静态的去给一些basis function去拟合上一个系数。就算你rolling的调整，也不好。因此真正想实现一个动态建模，就需要类似强化学习的一个data-driven way的方式去解决。文章使用的convolutional transformer也是一种data driven method。



**数据**
- 使用的是十九年的500个流动性最强的股票的日线数据，考虑了各种风险因子模型（比如Barra），对比了不同的金融数学里面的mean-reversion models (parametric/non-parametric)。结果是substantially outperforms all benchmark out-of-sample，年化夏普达到4，年化收益20%，波动率少于6%，同时收益和常规风险因子以及market movement无关（也就是不被宏观因子解释，属于是alpha部分的收益），同时扣除了交易手续费。同时stable over time and robust to tuning parameters，这个是我最关心的，对调参比较robust是大多数ML模型都能实现的，但是stable over time就很困难了。

**关于residual portfolio与factor loading**
- 文章强调了为什么用residual portfolio。我记得pre里面主持人就提问，她怀疑这个策略能否具体的implement，因为这个residual组合并不是很好交易，似乎作者给出的回答好像是使用ETF什么去近似，不是很理解。文章说用residual因为多数情况下，residual是stationary并且低相关，无论是时序上还是截面上。对这样的residual的return做时序建模就有意义，因为历史发生的事情未来也会发生，**而如果直接对日线return去用ML模型建模预测，基本不可能有好效果。这个也很好理解，直接对日线数据做什么时序建模，无论用什么时序模型GARCH什么ARIMA，他大概率不平稳，然后你又要用trend先去decompose，这一通操作乱七八糟的基本上结果就不会好，之前的两篇文章直接对日线的return拟合机器学习模型，基本表现稀烂，因为日线return就是由几个systematic factor在解释，时序上也可能具有一定的自相关性，而且由于行业板块的存在，轮动的存在，注定会有很大的heterogeneity，整体来说就是噪音太多不便于建模**。<font color=blue>所以这里用residual建模倒可以理解，不过我想如果是高频的日内数据，比如高频的期货数据，那么之前检验过基本上由于时间频率升高，受到的noise会小很多，因此之前CTA里面也测试过多数的时序是平稳的，然后间隔取降低频次，发现就是频次越高越不容易平稳，最后到4096个tick基本上对应半小时，就已经数据有些不平稳了，更不用说日线数据了。我们之前3080检验过daily return的一个分布情况，做过normal的分布检验，结论就是有很大的偏度和峰度，而且有很多outlier，拒绝假设检验，正态拒绝了，基本就不太可能是弱平稳了。那根据这个我想，这应该算是高频T0的一大优势所在，大量失效无用的日线时序模型、机器学习模型，也许在T0中就会有用。毕竟文章这个跟风险因子模型回归取残差，风险因子的计算值比如barra，比如fama-french，最高频也只有日度的吧，所以日内肯定是没法用的，但既然高频的log return序列本身就比较平稳的话，那是否可以跳过第一步直接尝试后面的步骤？</font>
- **这里和课内知识串起来了**！！！！<font color=blue>如果使用了风格因子做回归，那么就有了similarity的衡量，也就是对于risk factor的loading是相似的，这就说明两样资产的基本面信息，systematic的部分，beta的部分是相似的。那么这样构建多空组合，超额收益就是纯alpha，就是纯套利。这个完全符合课内讲的，2020课内讲反证法证明APT的时候，也是控制两个portfolio的对单因子/三因子的loading一样，然后做差证明存在套利机会，这本来才是课内讲的最正统的套利啊！包括3080我们APT那一章也考过通过系数配比，构建factor loading，验证是否有arbitrage的机会，一模一样！还有我们IVOL因子的检验，构建多空组合，也是通过调整对IVOL最小的组合和最大的组合的权重，使得这个配比权重下实现一个前面所说的market neutral，也就是整个多空组合针对market单因子或者fama三风险因子是中性的，然后做多空组合，那么收益就算纯alpha，算的就是纯因子收益的显著性。这下就把课内的知识，project的知识，和这篇论文串起来的，而且也把单因子检验、多空组合、中性策略和套利给串起来了。那这么看来，IVOL这个单因子的检验过程就是一个最正统的统计套利，我们在套一个real price和fair price的利，而这里fair price就是由多风险因子模型决定的。同时这么看来，前一篇文章的那个多空组合也算是套利，不过不是market/factor neutral，而是dollar neutral，文章也是这么说的，之前一直纠结这个算不算套利，因为不知道这种case下所谓的fair price是什么，现在看来就懂了，factor/neutral下的fair price是多因子模型算出来的，而dollar neutral下的fair price就是0，也就是多空组合的价值之差的fair值是0，如果不是0，我们会期待他回到0。所以在factor neutral下，我们对于多空组合的仓位分配是不同的，目的是为了凑factor loading一样，但dollar neutral下，我们对多空组合的仓位分配是相同的。</font>
- 那这么说，这种最正统的factor neutral的套利只能对日线数据使用，而且只能是股票市场哈哈。而dollar neutral虽然不算，但日内高频的套利能用啊。所以文章使用因子模型的loading作为similarity的衡量是正确的，目的就是通过系数配比，把long-short portfolio对因子的loading抵消掉。

现在的疑问就是，怎么选套利的组合？loading不可能完全一样，难道就几个loading去算correlation？以及谁多谁空？是回归常数项为正的就多，为负的就空吗？


**模型**
还没看完


#### 1.10 关于套利的思考与总结

如果让我把实习和之前了解到的所有套利相关的理论、概念，以及阅读过的统计套利的文献，做的leadlag研究等全部总结成一个大的PPT去present，会怎么去讲？

- 关于套利的基础理论：FIN2020的内容，设计套利的部分，arrow-debru理论，martingale理论，**APT模型与多因子模型**

- pair trading
- 套利的必须要素
- 可能有什么套利策略？多空组合，结合论文、论文presentation的PPT、点宽对于CTA套利的讲解，babyquant的CTA的套利部分。
- 币圈的套利（期现套利、跨期套利、跨交易所套利），最好部署实盘的套利策略？
- 复杂的机器学习和深度学习怎么帮助套利？



### 2. 经典套利策略回测


#### 2.1 复现crypto arbitrage文章的策略
目前看的文献当中, 最容易implement的就是statistical arbitrage in cryptocurrency market的策略, 也是美股那一篇已经证明过无效的策略。但是当时毕竟使用的是日线数据, 而且作为最经典的套利策略, 有必要去复现一下框架, 而且里面用到的ML模型也可以和CTA部分的ML/DL模型呼应, 形成相互促进的关系。而且本文github有复现的代码但没有数据, 因此首先要复现的最适合高频的最简单的套利策略就是这个。 