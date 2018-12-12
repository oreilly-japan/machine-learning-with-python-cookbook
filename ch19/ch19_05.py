# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# データをロード
iris = datasets.load_iris()
features = iris.data

# 特徴量を標準化
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# 凝集型クラスタリング器を作成
cluster = AgglomerativeClustering(n_clusters=3)

# 凝集型クラスタリング器を訓練
model = cluster.fit(features_std)

###########

# クラスタ割当を表示
model.labels_
