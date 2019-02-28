"""
Feature
"""

import numpy as np
import pandas as pd
import datetime
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, RepeatedKFold
from sklearn.metrics import mean_squared_error

# train = pd.read_csv("F:\\Mycode\\Kaggle\\Kaggle\\Elo\\data\\train.csv")
# test = pd.read_csv("F:\\Mycode\\Kaggle\\Kaggle\\Elo\\data\\test.csv")
# test_card_id = test[['card_id']]
historical_transactions = pd.read_csv("F:\\Mycode\\Kaggle\\Kaggle\\Elo\\data\\historical_transactions.csv")
train = pd.read_csv("F:\\Mycode\\Kaggle\\Kaggle\\Elo\\data\\train_merge.csv")
test = pd.read_csv("F:\\Mycode\\Kaggle\\Kaggle\\Elo\\data\\test_merge.csv")
# new_merchant_transactions = pd.read_csv("F:\\Mycode\\Kaggle\\Kaggle\\Elo\\data\\new_merchant_transactions.csv")


# 异常值处理, 异常值2207
train['outliers'] = 0
train.loc[train['target'] < -30, 'outliers'] = 1

# 天数
# 转换first_active_month为时间格式
train['first_active_month'] = pd.to_datetime(train['first_active_month'])
test['first_active_month'] = pd.to_datetime(test['first_active_month'])

train['days'] = (datetime.date(2018, 2, 1) - train['first_active_month'].dt.date).dt.days
feature_cols = ['feature_1', 'feature_2', 'feature_3']
for f in feature_cols:
    train['days_' + f] = train['days'] * train[f]
    train['days_' + f + '_ratio'] = train[f] / train['days']

test['days'] = (datetime.date(2018, 2, 1) - test['first_active_month'].dt.date).dt.days
feature_cols = ['feature_1', 'feature_2', 'feature_3']
for f in feature_cols:
    test['days_' + f] = test['days'] * test[f]
    test['days_' + f + '_ratio'] = test[f] / test['days']

"""
new_merchant_transactions数据处理
"""
# 缺失值处理
historical_transactions['category_2'].fillna(1.0, inplace=True)
historical_transactions['category_3'].fillna('A', inplace=True)
historical_transactions['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
historical_transactions['installments'].replace(-1, np.nan, inplace=True)
historical_transactions['purchase_amount'] = historical_transactions['purchase_amount'].apply(lambda x: min(x, 0.8))

# 特征工程
historical_transactions['authorized_flag'] = historical_transactions['authorized_flag'].map({'Y': 1, 'N': 0})
historical_transactions['category_1'] = historical_transactions['category_1'].map({'Y': 1, 'N': 0})
historical_transactions['category_3'] = historical_transactions['category_3'].map({'A': 1, 'B': 2, 'C': 3})

historical_transactions['purchase_date'] = pd.to_datetime(historical_transactions['purchase_date'])
historical_transactions['month'] = historical_transactions['purchase_date'].dt.month
historical_transactions['weekofyear'] = historical_transactions['purchase_date'].dt.weekofyear
historical_transactions['day'] = historical_transactions['purchase_date'].dt.day
historical_transactions['weekday'] = historical_transactions['purchase_date'].dt.weekday
historical_transactions['weekend'] = (historical_transactions['purchase_date'].dt.weekday > 5).astype(int)
historical_transactions['hour'] = historical_transactions['purchase_date'].dt.hour
historical_transactions['month_diff'] = ((datetime.datetime.today() - historical_transactions['purchase_date']).dt.days) // 30
historical_transactions['month_diff'] += historical_transactions['month_lag']

historical_transactions['duration'] = historical_transactions['purchase_amount'] * historical_transactions['month_diff']
historical_transactions['amount_month_ratio'] = historical_transactions['purchase_amount'] / historical_transactions['month_diff']
historical_transactions['price'] = historical_transactions['purchase_amount'] / historical_transactions['installments']

for col in ['category_2', 'category_3']:
    historical_transactions[col + '_mean'] = historical_transactions['purchase_amount'].groupby(
        historical_transactions[col]).agg('mean')
    historical_transactions[col + '_max'] = historical_transactions['purchase_amount'].groupby(
        historical_transactions[col]).agg('max')
    historical_transactions[col + '_min'] = historical_transactions['purchase_amount'].groupby(
        historical_transactions[col]).agg('min')
    historical_transactions[col + '_sum'] = historical_transactions['purchase_amount'].groupby(
        historical_transactions[col]).agg('sum')

# aggregate函数
def aggregate_historical_transactions(transactions, prefix):
    agg_function = {
        'purchase_amount': ['sum', 'max', 'min', 'mean', 'var', 'skew'],
        'installments': ['sum', 'max', 'mean', 'var', 'skew'],
        'purchase_date': ['max', 'min'],
        'month_lag': ['max', 'min', 'mean', 'var', 'skew'],
        'month_diff': ['max', 'min', 'mean', 'var', 'skew'],
        'weekend': ['sum', 'mean'],
        'weekday': ['sum', 'mean'],
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['sum', 'mean', 'max', 'min'],
        'card_id': ['size', 'count'],
        'month': ['nunique', 'mean', 'min', 'max'],
        'hour': ['nunique', 'mean', 'min', 'max'],
        'weekofyear': ['nunique', 'mean', 'min', 'max'],
        'day': ['nunique', 'mean', 'min', 'max'],
        'subsector_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'price': ['sum', 'mean', 'max', 'min', 'var'],
        'duration': ['mean', 'min', 'max', 'var', 'skew'],
        'amount_month_ratio': ['mean', 'min', 'max', 'var', 'skew']
    }
    agg_transactions = transactions.groupby(['card_id']).agg(agg_function)
    agg_transactions.columns = [prefix + '_'.join(col).strip() for col in agg_transactions.columns.values]
    agg_transactions.reset_index(inplace=True)
    df = (transactions.groupby(['card_id']).size().reset_index(name='{}transactions_count'.format(prefix)))
    agg_transactions = pd.merge(df, agg_transactions, on='card_id', how='left')

    return agg_transactions

"""
historical_transactions处理数据
"""
# 缺失值处理
historical_transactions['category_2'].fillna(1.0, inplace=True)
historical_transactions['category_3'].fillna('A', inplace=True)
historical_transactions['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
historical_transactions['installments'].replace(-1, np.nan, inplace=True)
historical_transactions['purchase_amount'] = historical_transactions['purchase_amount'].apply(lambda x: min(x, 0.8))

# 特征工程
historical_transactions['authorized_flag'] = historical_transactions['authorized_flag'].map({'Y': 1, 'N': 0})
historical_transactions['category_1'] = historical_transactions['category_1'].map({'Y': 1, 'N': 0})
historical_transactions['category_3'] = historical_transactions['category_3'].map({'A': 1, 'B': 2, 'C': 3})

historical_transactions['purchase_date'] = pd.to_datetime(historical_transactions['purchase_date'])
historical_transactions['month'] = historical_transactions['purchase_date'].dt.month
historical_transactions['weekofyear'] = historical_transactions['purchase_date'].dt.weekofyear
historical_transactions['day'] = historical_transactions['purchase_date'].dt.day
historical_transactions['weekday'] = historical_transactions['purchase_date'].dt.weekday
historical_transactions['weekend'] = (historical_transactions['purchase_date'].dt.weekday > 5).astype(int)
historical_transactions['hour'] = historical_transactions['purchase_date'].dt.hour
historical_transactions['month_diff'] = ((datetime.datetime.today() - historical_transactions['purchase_date']).dt.days) // 30
historical_transactions['month_diff'] += historical_transactions['month_lag']

historical_transactions['duration'] = historical_transactions['purchase_amount'] * historical_transactions['month_diff']
historical_transactions['amount_month_ratio'] = historical_transactions['purchase_amount'] / historical_transactions['month_diff']
historical_transactions['price'] = historical_transactions['purchase_amount'] / historical_transactions['installments']

for col in ['category_2', 'category_3']:
    historical_transactions[col + '_mean'] = historical_transactions['purchase_amount'].groupby(
        historical_transactions[col]).agg('mean')
    historical_transactions[col + '_max'] = historical_transactions['purchase_amount'].groupby(
        historical_transactions[col]).agg('max')
    historical_transactions[col + '_min'] = historical_transactions['purchase_amount'].groupby(
        historical_transactions[col]).agg('min')
    historical_transactions[col + '_sum'] = historical_transactions['purchase_amount'].groupby(
        historical_transactions[col]).agg('sum')




historical_transactions = aggregate_historical_transactions(historical_transactions, prefix='his_')
train = pd.merge(train, historical_transactions, on='card_id', how='left')
test = pd.merge(test, historical_transactions, on='card_id', how='left')
train.to_csv("F:\\Mycode\\Kaggle\\Kaggle\\Elo\\data\\train_whole.csv")
test.to_csv("F:\\Mycode\\Kaggle\\Kaggle\\Elo\\data\\test_whole.csv")
