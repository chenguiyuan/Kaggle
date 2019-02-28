"""
This is project to Elo of Kaggle
"""

import numpy as np
import pandas as pd
import datetime
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, RepeatedKFold
from sklearn.metrics import mean_squared_error

train = pd.read_csv("F:\\Mycode\\Kaggle\\Kaggle\\Elo\\data\\train_whole.csv")
test = pd.read_csv("F:\\Mycode\\Kaggle\\Kaggle\\Elo\\data\\test_whole.csv")

target = train[['target']]
submission = test[['card_id']]
train = train.drop(['card_id', 'first_active_month', 'target', 'new_price_sum', 'new_price_mean', 'new_price_max', 'new_price_min', 'his_purchase_date_max', 'his_purchase_date_min', 'new_purchase_date_max', 'new_purchase_date_min'], axis=1)
test = test.drop(['card_id', 'first_active_month', 'new_price_sum', 'new_price_mean', 'new_price_max', 'new_price_min', 'his_purchase_date_max', 'his_purchase_date_min', 'new_purchase_date_max', 'new_purchase_date_min'], axis=1)

folds = RepeatedKFold(n_splits=5, random_state=4950)


features = [c for c in train.columns if c not in ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size']]

for fold_, (train_idx, validate_idx) in enumerate(folds.split(train.values, target.values)):

    # train_data = lgb.Dataset(train.iloc[train_idx][features], target.iloc[train_idx])
    # validate_data = lgb.Dataset(train.iloc[validate_idx][features], target.iloc[validate_idx])
    train_data = lgb.Dataset(train.iloc[train_idx][features], label=target.iloc[train_idx])
    validate_data = lgb.Dataset(train.iloc[validate_idx][features], label=target.iloc[validate_idx])
    param = {
            'task': 'train',
            'boosting': 'goss',
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'subsample': 0.9855232997390695,
            'max_depth': 7,
            'top_rate': 0.9064148448434349,
            'num_leaves': 90,#63
            'min_child_weight': 41.9612869171337,
            'other_rate': 0.0721768246018207,
            'reg_alpha': 9.677537745007898,
            'colsample_bytree': 0.5665320670155495,
            'min_split_gain': 9.820197773625843,
            'reg_lambda': 8.2532317400459,
            'min_data_in_leaf': 21,
            'verbose': -1,
            'seed': int(2 ** fold_),
            'bagging_seed': int(2 ** fold_),
            'drop_seed': int(2 ** fold_)
            }
    num_round = 1000
    clf = lgb.train(param, train_data, num_round, valid_sets=[train_data, validate_data], verbose_eval=-1, early_stopping_rounds=200)
    # oof[validate_idx] = clf.predict(train.iloc[validate_idx], num_iteration=200)
    # predictions['target'] = clf.predict(test, num_iteration=50)

submission['target'] = clf.predict(test, num_iteration=100)
submission.reset_index(drop=True)
submission.to_csv("F:\\Mycode\\Kaggle\\Kaggle\\Elo\\data\\submission0222.csv")

"""
GridSearchCV
"""

# from sklearn.model_selection import GridSearchCV
#
# param = {
#             'learning_rate': [0.001, 0.01, 0.02, 0.1],
#             'max_depth': [5, 7, 9],
#             'num_leaves': [30, 63, 90]
#             }
# model = lgb.LGBMRegressor(objective='regression', boosting_type='goss', subsample=0.9855232997390695,
#                           min_child_weight=41.9612869171337, reg_alpha=9.677537745007898, colsample_bytree=0.5665320670155495,
#                           min_split_gain=9.820197773625843, reg_lambda=8.2532317400459)
# gsearch = GridSearchCV(estimator=model, param_grid=param, scoring='neg_mean_squared_error')
# gsearch.fit(train, target)
# print(gsearch.best_params_)
# print(gsearch.best_score_)