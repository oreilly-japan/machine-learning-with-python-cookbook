# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

# データをロード
iris = datasets.load_iris()
features = iris.data

# 特徴量を標準化
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# k-平均法クラスタリング器を作成
cluster = MiniBatchKMeans(n_clusters=3, random_state=0, batch_size=100)

# k-平均法クラスタリング器を訓練
model = cluster.fit(features_std)


