# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift

# データをロード
iris = datasets.load_iris()
features = iris.data

# 特徴量を標準化
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# 平均値シフト法クラスタリング器を作成
cluster = MeanShift(n_jobs=-1)

# 平均値シフト法クラスタリング器を訓練
model = cluster.fit(features_std)
