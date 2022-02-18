# -*- coding:utf-8 -*-
"""
第4章：数据清洗与预处理
"""
import os
import pandas as pd
import numpy as np
import missingno as msno
import warnings
import matplotlib.pyplot as plt
import matplotlib

# 用黑体显示中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 正常显示负号
matplotlib.rcParams['axes.unicode_minus'] = False
# 忽略警告
warnings.filterwarnings("ignore")
# 设置显示多列
pd.set_option('display.max_columns', 10)
# 设置全部显示
pd.set_option('display.max_columns', None)


# 读取数据
def data_read(data_path, file_name):
    df = pd.read_csv(
        os.path.join(
            data_path,
            file_name),
        delim_whitespace=True,
        header=None)
    columns = [
        'status_account',
        'duration',
        'credit_history',
        'purpose',
        'amount',
        'svaing_account',
        'present_emp',
        'income_rate',
        'presonal_status',
        'other_debtors',
        'residence_info',
        'property',
        'age',
        'inst_plans',
        'housing',
        'num_credits',
        'job',
        'dependents',
        'telephone',
        'foreign_worker',
        'target']
    df.columns = columns
    df.target = df.target - 1
    return df


# 划分离散变量和连续变量
def category_continue_separation(df):
    feature_names = list(df.columns)
    feature_names.remove('target')
    numerical_list = list(
        df[feature_names].select_dtypes(
            include=[
                'int',
                'float',
                'int32',
                'float32',
                'int64',
                'float64']).columns.values)
    categorical_list = [x for x in feature_names if x not in numerical_list]
    return categorical_list, numerical_list


# 查看离散变量清洗特殊字符
def category_clean(df, category_list):
    for var in category_list:
        print(df[var].unique())
        df[var] = df[var].apply(lambda x: x.replace(' ', '').replace('%', ''))
    return df


# 样本去冗余
def drop_duplicate_sample(df):
    # 剔除完全相同的样本
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    # 按一个变量或多个变量去冗余
    # df.drop_duplicates(subset=['order_id'], keep='first', inplace=True)
    # 剔除重复的列名
    # df_1 = df.T
    # df_1 = df_1[~df_1.index.duplicated()]
    # df = df_1.T
    return df


# 缺失值分析
def missing_value(df):
    msno.bar(df, labels=True, figsize=(10, 6), fontsize=10)
    # plt.show()


# 异常值分析
def outliers(df, numerical_var):
    plt.figure(figsize=(10, 6))  # 设置图形尺寸大小
    for j in range(1, len(numerical_var) + 1):
        plt.subplot(2, 4, j)
        df_temp = df[numerical_var[j - 1]][~df[numerical_var[j - 1]].isnull()]
        plt.boxplot(df_temp,
                    notch=False,  # 中位线处不设置凹陷
                    widths=0.2,  # 设置箱体宽度
                    medianprops={'color': 'red'},  # 中位线设置为红色
                    boxprops=dict(color="blue"),  # 箱体边框设置为蓝色
                    labels=[numerical_var[j - 1]],  # 设置标签
                    whiskerprops={'color': "black"},  # 设置须的颜色，黑色
                    capprops={'color': "green"},  # 设置箱线图顶端和末端横线的属性，颜色为绿色
                    flierprops={
                        'color': 'purple',
                        'markeredgecolor': "purple"}  # 异常值属性，这里没有异常值，所以没表现出来
                    )
    # plt.show()


# 连续变量不同类别下的分布
def numerical_distribution(df, numerical_var, path):

    for i in numerical_var:
        # 取非缺失值的数据
        df_temp = df.loc[~df[i].isnull(), [i, 'target']]
        df_good = df_temp[df_temp.target == 0]
        df_bad = df_temp[df_temp.target == 1]
        # 计算统计量
        valid_ = round(df_temp.shape[0] / df.shape[0] * 100, 2)
        mean_ = round(df_temp[i].mean(), 2)
        std_ = round(df_temp[i].std(), 2)
        max_ = round(df_temp[i].max(), 2)
        min_ = round(df_temp[i].min(), 2)
        # 绘图
        plt.figure(figsize=(10, 6))
        fontsize_1 = 12
        plt.hist(df_good[i], bins=20, alpha=0.5, label='好样本')
        plt.hist(df_bad[i], bins=20, alpha=0.5, label='坏样本')
        plt.ylabel(i, fontsize=fontsize_1)
        plt.title(
            'valid rate=' +
            str(valid_) +
            '%, Mean=' +
            str(mean_) +
            ', Std=' +
            str(std_) +
            ', Max=' +
            str(max_) +
            ', Min=' +
            str(min_))
        plt.legend()
        # 保存图片
        file = os.path.join(path, 'plot_num', i + '.png')
        plt.savefig(file)
        plt.close(1)


# 离散变量不同类别下的分布
def categorical_distribution(df, categorical_var, path):

    for i in categorical_var:
        # 非缺失值数据
        df_temp = df.loc[~df[i].isnull(), [i, 'target']]
        df_bad = df_temp[df_temp.target == 1]
        valid = round(df_temp.shape[0] / df.shape[0] * 100, 2)

        bad_rate = []
        bin_rate = []
        var_name = []
        for j in df[i].unique():

            if pd.isnull(j):
                df_1 = df[df[i].isnull()]
                bad_rate.append(sum(df_1.target) / df_1.shape[0])
                bin_rate.append(df_1.shape[0] / df.shape[0])
                var_name.append('NA')
            else:
                df_1 = df[df[i] == j]
                bad_rate.append(sum(df_1.target) / df_1.shape[0])
                bin_rate.append(df_1.shape[0] / df.shape[0])
                var_name.append(j)
        df_2 = pd.DataFrame(
            {'var_name': var_name, 'bin_rate': bin_rate, 'bad_rate': bad_rate})
        # 绘图
        plt.figure(figsize=(10, 6))
        fontsize_1 = 12
        plt.bar(
            np.arange(
                1,
                df_2.shape[0] + 1),
            df_2.bin_rate,
            0.1,
            color='black',
            alpha=0.5,
            label='取值占比')
        plt.xticks(np.arange(1, df_2.shape[0] + 1), df_2.var_name)
        plt.plot(
            np.arange(
                1,
                df_2.shape[0] + 1),
            df_2.bad_rate,
            color='green',
            alpha=0.5,
            label='坏样本比率')

        plt.ylabel(i, fontsize=fontsize_1)
        plt.title('valid rate=' + str(valid) + '%')
        plt.legend()
        # 保存图片
        file = os.path.join(path, 'plot_cat', i + '.png')
        plt.savefig(file)
        plt.close(1)


if __name__ == '__main__':
    # 读取数据
    df = data_read('../data/input/', 'german.csv')
    # 划分离散变量和连续变量
    category_continue_separation(df)
    categorical_var, numerical_var = category_continue_separation(df)
    # 离散变量剔除特殊字符
    df = category_clean(df, categorical_var)
    # 样本去冗余
    df = drop_duplicate_sample(df)
    # 缺失值处理
    missing_value(df)
    # 异常值处理
    outliers(df, numerical_var)
    # 变量分布分析
    numerical_distribution(df, numerical_var, '../data/output/')
    categorical_distribution(df, categorical_var, '../data/output/')


# 特殊字符定位
char_list = [' ','?','@','#','$','/t','*',';','&']
unknowns = {}
df['policy_number']
for i in list(df.columns):
    if (df[i]).dtype == object:
        j = np.sum(df[i] == "?")
        unknowns[i] = j
unknowns = pd.DataFrame.from_dict(unknowns, orient = 'index')
print(unknowns)