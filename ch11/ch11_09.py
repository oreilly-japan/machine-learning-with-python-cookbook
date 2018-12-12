# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 特徴量行列を作成
features, _ = make_blobs(n_samples = 1000,
                         n_features = 10,
                         centers = 2,
                         cluster_std = 0.5,
                         shuffle = True,
                         random_state = 1)

# k-meansを用いてデータをクラスタリングしクラスを予想
model = KMeans(n_clusters=2, random_state=1).fit(features)

# 予想されたクラスを取得
target_predicted = model.labels_

# モデルを評価
silhouette_score(features, target_predicted)

