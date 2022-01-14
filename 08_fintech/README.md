# 金融大数据风控建模实战


## 一、金融科技
### （1）智能风控和评分卡

智能风控基于“大数据+人工智能技术”简历信用评估模型，提高风险管控能力。在金融领域，最核心的问题是风险控制。

个人信贷中信用风险评估的关键是，通过分析借款人的信用数据和其他相关数据，评估借款人的还款意愿和能力，从而量化违约风险。

评分卡模型包括：

- 申请评分卡
- 行为评分卡
- 催收评分卡
- 营销评分卡
- 客户流式评分卡
- 反欺诈模型

注：逾期与违约的含义不同，违约是严重的逾期行为，所以逾期用户一定是从违约发展而来的。


### （1）数据准备

#### 数据格式

- txt
- csv
- excel
- json
- xml

#### 文件读写（file_help）
```
pd.read_csv( )
pd.to_csv( )
```
#### 数据库读写（db_help）


### （2）数据探索

#### 查看数据整体信息
```
# 查看数据形状
data.shape
# 查看数据基本信息
data.info()
# 查看数据前5条+后5条
data.head().append(data.tail())
# 查看数据的统计情况
data.describe()
```
#### 查看数据缺失情况

- 含有缺失值的列的个数：
```
missing = missing[missing > 0]
```
- 每个列的缺失比例：
```
data.isnull().sum()/data.isnull().count()
```
- 可视化缺失情况

#### 查看数据异常情况

#### 查看样本标签分布情况

- 查看连续性标签的情况：
```
data.describe()
```
- 查看离散型标签的情况：
```
print(pd.Series(Y_data).value_counts())
```
- 可视化标签分布情况
- 特征类型分析

#### 数字特征分析

- 获取数值型特征
```
numerical_fea = data_train.select_dtypes(exclude=['object']).columns
```
- 数值离散型变量分析
- 类别分布
```
data_train['term'].value_counts()
```
- 类别数
```
data_train['term'].nunique()
```
- 数值连续型变量分析
- 每个数字特征的分布可视化
- 相关性分析
- 特征之间的相关性可视化
#### 类型特征分析
- 获取类别型特征
```
category_fea = data_train.select_dtypes(include=['object']).columns
```
- 类别数统计
- 类别特征分布情况

#### 查看特征变量和目标变量的相关系数

#### 用pandas_profiling生成数据报告

#### 数据可视化
- 直方图
- 条形图
- 散点图
- 箱型图
- 热力图
- 折线图
- 饼图
- QQ图
- 蜘蛛图
- 成对关系图

### （3）特征工程

#### 数据清洗

- 缺失值处理
- 异常值处理
- 去除噪声
- 降维
- 其他
​				删除无效列
​				更改dtypes
​				删除列中的字符串
​				字符串转时间

#### 特征构造

- 统计量特征：计数、求和、比例、标准差
- 时间特征：绝对时间、相对时间、节假日、双休日
- 地理信息：分桶
- 非线性变换：取log/平方/根号
- 数据分桶：等频/等距分桶、Best-KS分桶、卡方分桶
- 特征组合/特征交叉

#### 特征变换

- 标准化：统一量纲，取值范围不限
- 归一化：映射到0-1，取值范围[0,1]
- 二值化
- 定性变量编码：onehot encode等

#### 特征选择

- IV值
- 相关性分析
- 特征重要性评分


```
处理类别不平衡的小技巧：
​		扩充数据集
​		尝试其他评估指标，AUC等
​		调整θ值
​		重采样：过采样/欠采样
​		合成样本，SMOTE
​		加权少类别的样本错分代价
```


### （4）数据建模

### （5）模型评估

#### 评估方法

- 交叉验证法
- 留一法
- 自助法

#### 评估指标

- precision
- recall
- acc
- MAE
- MSE

### （6）调参优化

#### 手动调参

- 提高准确率
- 防止过拟合
- 提高训练速度

#### 随机搜索

#### 网格搜索

#### 贝叶斯调参

### （7）模型部署

#### API

- web service
- restful

#### python脚本

#### 可执行程序

## 二、常用算法

### （1）有监督学习（分类、回归）

- 朴素贝叶斯

- 线性回归

- 逻辑回归

- SVM

- KNN

- GBDT

- 决策树

- 随机森林

- Adaboost

- Catboost

- Xgboost

- LightGBM

- 关联规则（Apriori）



### （2）无监督学习（聚类、降维）

#### 聚类

- k均值聚类（K-means）
- 层次聚类（Hierarchical clustering）
- DBSCAN

#### 降维

- 线性判别分析（LinearDiscriminantAnalysis）
- 主成分分析（PCA）

### （3）重点概念
- bagging
- boosting
- CART
- ID3
- C4.5
- sigmoid
- softmax
- bias
- variance
- accuracy
- precision
- recall
- f1-score
- roc
- auc
- ks
- tpr
- fpr
- mae
- mse
- rmse
- r²
- mape
- underfitting
- overfitting
