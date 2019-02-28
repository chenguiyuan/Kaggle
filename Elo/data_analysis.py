"""
data analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import datetime as dt
from datetime import datetime
import time

train = pd.read_csv("F:\\Kaggle\\Elo\\data\\train.csv")
test = pd.read_csv("F:\\Kaggle\\Elo\\data\\test.csv")
# print(test.info())
# print(train.info())
# print(train['feature_2'].value_counts())
# print(train['feature_3'].value_counts())
# print(train['first_active_month'].value_counts())

# 柱状图feature_1, feature_2, feature_3
# plt.figure()
# # fig1 = train['feature_1'].value_counts().sort_index()
# # fig1.plot(kind='bar', title='feature_1', color='blue')
# fig2 = train['feature_2'].value_counts().sort_index()
# fig2.plot(kind='bar', title='feature_2', color='red')
# # fig3 = train['feature_3'].value_counts().sort_index()
# # fig3.plot(kind='bar', title='feature_3', color='green')
# plt.show()

# 分类数据可视化
# fig, ax = plt.subplot(1, 3, figsize=(16, 6))
# sns.violinplot(x='feature_1', y='target', ax=ax[0], data=train, title='feature_1')
# sns.violinplot(x='feature_2', y='target', ax=ax[1], data=train, title='feature_2')
# sns.violinplot(x='feature_3', y='target', ax=ax[2], data=train, title='feature_3')
# plt.show()

# 统计每月首次激活的用户数
# train['first_active_month'] = pd.to_datetime(train['first_active_month'])
# test['first_active_month'] = pd.to_datetime(test['first_active_month'])
# # fig = plt.subplot(figsize=(15, 6))
# d1 = train.groupby('first_active_month').count()['card_id']
# d2 = test.groupby('first_active_month').count()['card_id']
# d1.plot(color='r')
# d2.plot(color='b')
# plt.title('first_activate_month')
# plt.show()

# 对单个属性绘制直方图，给定值范围，查看实例数量
# train['target'].plot(kind='hist', bins=50, title='Target_distrubution')
# plt.show()

# 相关性系数
# corr_martix = train.corr()
# corr_martix['target'].sort_values(ascending=False)
# print(corr_martix)

# 获取最大时间
# max_date = datetime.date(train['first_active_month'])

train['first_active_month'] = pd.to_datetime(train['first_active_month'])
max_data = train['first_active_month'].max()
print(max_data)
# test['first_active_month'] = pd.to_datetime(test['first_active_month'])

# 特征处理，年/月/日/周/星期/时间间隔
# combine = [train, test]
# for df in combine:
#     df['year'] = df['first_active_month'].year()
# #     df['month'] = df['first_active_month'].month()
# #     df['week'] = df['first_active_month'].weekofyear()
# #     df['dayofweek'] = df['first_active_month'].dayofweek()
# #     df['period'] = (max_data - df['first_active_month']).days
# # target = train['target']
# # del train['target']
# # print(train.head())
# for df in train:
#     df['year'] = df['first_active_month'][:4]
# train['year'] = train['first_active_month'].str[:4]
# print(train['year'])