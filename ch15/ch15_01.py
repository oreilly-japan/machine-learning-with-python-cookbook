# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# データをロード
iris = datasets.load_iris()
features = iris.data

# 標準化器を作成
standardizer = StandardScaler()

# 特徴量を標準化
features_standardized = standardizer.fit_transform(features)

# 2-最近傍法
nearest_neighbors = NearestNeighbors(n_neighbors=2).fit(features_standardized)

# 観測値を作成
new_observation = [ 1,  1,  1,  1]

# 最近傍の観測値の距離とインデックスを計算
distances, indices = nearest_neighbors.kneighbors([new_observation])

# 最近傍点を表示
features_standardized[indices]

##########

# ユークリッド距離を用いて最初の2つの最近傍点を見つける
nearestneighbors_euclidean = NearestNeighbors(
n_neighbors=2, metric='euclidean').fit(features_standardized)

##########

# 距離を表示
distances

##########

# ユークリッド距離を用いて、最近傍点を3つ(自身を含む)見つける
nearestneighbors_euclidean = NearestNeighbors(
n_neighbors=3, metric="euclidean").fit(features_standardized)

# それぞれの観測値に対する3つの最近傍点(自身を含む)
# を表すリストのリストを作成
nearest_neighbors_with_self = nearestneighbors_euclidean.kneighbors_graph(
    features_standardized).toarray()

# 自分自身を指している近傍点を削除
for i, x in enumerate(nearest_neighbors_with_self):
    x[i] = 0

# 最初の観測値の2つの最近傍点を表示
nearest_neighbors_with_self[0]




