# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.linear_model import LogisticRegressionCV
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 特徴量を標準化
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# ロジスティック回帰器を作成
logistic_regression = LogisticRegressionCV(
    penalty='l2', Cs=10, random_state=0, n_jobs=-1)

# ロジスティック回帰器を訓練
model = logistic_regression.fit(features_standardized, target)

