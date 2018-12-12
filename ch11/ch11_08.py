# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# ターゲットベクトルと特徴量行列を作成
features, target = make_regression(n_samples = 100,
                                   n_features = 3,
                                   n_informative = 3,
                                   n_targets = 1,
                                   noise = 50,
                                   coef = False,
                                   random_state = 1)

# 線形回帰器を作成
ols = LinearRegression()

# MSE(の符号を反転したもの)を用いて線形回帰器を交差検証
cross_val_score(ols, features, target, scoring='neg_mean_squared_error')

##########

# R^2を用いて線形回帰器を交差検証
cross_val_score(ols, features, target, scoring='r2')
