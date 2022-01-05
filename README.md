# 机器学习

机器学习是一门多领域交叉学科，专门研究计算机怎样模拟或实现人类的学习行为，以获取新的知识或技能，重新组织已有的知识结构使之不断改善自身的性能。

![](https://github.com/WengSongxiu/MachineLearning/blob/master/image/ml.jpg)


## 一、整体流程
机器学习项目的整体流程可以大致分为以下几个步骤：

![](https://github.com/WengSongxiu/MachineLearning/blob/master/image/process.png)


### （1）数据准备

#### 数据格式

- txt
- csv
- excel
- json
- xml

#### 文件读写（file_help）

- pd.read_csv( )
- pd.to_csv( )

#### 数据库读写（db_help）


### （2）数据探索

#### 查看数据整体信息

- data.shape
- data.info()
- data.head().append(data.tail())
- data.describe()

#### 查看数据缺失情况

- 含有缺失值的列的个数：missing = missing[missing > 0]
- 每个列的缺失比例：data.isnull().sum()/data.isnull().count()
- 可视化缺失情况

#### 查看数据异常情况

#### 查看样本标签分布情况

- 查看连续性标签的情况：data.describe()
- 查看离散型标签的情况：print(pd.Series(Y_data).value_counts())
- 可视化标签分布情况
- 特征类型分析

#### 数字特征分析

- 获取数值型特征numerical_fea = data_train.select_dtypes(exclude=['object']).columns
- 数值离散型变量分析
- 类别分布data_train['term'].value_counts()
- 类别数data_train['term'].nunique()
- 数值连续型变量分析
- 每个数字特征的分布可视化
- 相关性分析
- 特征之间的相关性可视化
#### 类型特征分析
- 获取类别型特征category_fea = data_train.select_dtypes(include=['object']).columns
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
​		类别不平衡
​			扩充数据集
​			尝试其他评估指标，AUC等
​			调整θ值
​			重采样：过采样/欠采样
​			合成样本，SMOTE
​			加权少类别的样本错分代价


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
- - rmse
- r²
- mape
- underfitting
- overfitting
