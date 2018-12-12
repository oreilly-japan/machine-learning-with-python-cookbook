# -*- coding: utf-8 -*-

# ライブラリをロード
import warnings
from sklearn.datasets import make_regression
from sklearn.feature_selection import RFECV
from sklearn import datasets, linear_model

# わずらわしいが無害な警告を抑止
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")

# 特徴量行列、ターゲットベクトル、真の相関係数を作成
features, target = make_regression(n_samples = 10000,
                                   n_features = 100,
                                   n_informative = 2,
                                   random_state = 1)

# 線形回帰器を作成
ols = linear_model.LinearRegression()

# 再帰的に特徴量を除去
rfecv = RFECV(estimator=ols, step=1, scoring="neg_mean_squared_error")
rfecv.fit(features, target)
rfecv.transform(features)

##########

# 最良の特徴量の数
rfecv.n_features_

##########

# 最良のカテゴリを表示
rfecv.support_
