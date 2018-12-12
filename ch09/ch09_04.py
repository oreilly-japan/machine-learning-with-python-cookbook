# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.decomposition import NMF
from sklearn import datasets

# データをロード
digits = datasets.load_digits()

# データを特徴量行列として使用
features = digits.data

# NMFを作成、訓練して適用
nmf = NMF(n_components=10, random_state=1)
features_nmf = nmf.fit_transform(features)

# 結果を表示
print("もとの特徴量数:", features.shape[1])
print("削減後の特徴量数:", features_nmf.shape[1])
