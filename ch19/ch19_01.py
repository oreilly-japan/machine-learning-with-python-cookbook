# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# データをロード
iris = datasets.load_iris()
features = iris.data

# 特徴量を標準化
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# k-平均法クラスタリングを作成
cluster = KMeans(n_clusters=3, random_state=0, n_jobs=-1)

# k-平均法クラスタリングを訓練
model = cluster.fit(features_std)

###########

# クラス予測結果を表示
model.labels_

###########

# クラスの真の値を表示
iris.target

###########

# 新たな観測値を作成
new_observation = [[0.8, 0.8, 0.8, 0.8]]

# 観測値のクラスタを予測
model.predict(new_observation)

###########

# クラスタ中心を表示
model.cluster_centers_
