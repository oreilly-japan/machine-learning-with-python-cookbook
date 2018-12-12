# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 特徴量を標準化
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# 1対その他ロジスティック回帰器を作成
logistic_regression = LogisticRegression(random_state=0, multi_class="ovr")

# ロジスティック回帰器を訓練
model = logistic_regression.fit(features_standardized, target)

