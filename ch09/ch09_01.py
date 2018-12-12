# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets

# データをロード
digits = datasets.load_digits()

# 特徴量行列を標準化
features = StandardScaler().fit_transform(digits.data)

# 分散を99%維持するPCAを作成
pca = PCA(n_components=0.99, whiten=True)

# PCAを実行
features_pca = pca.fit_transform(features)

# 結果を表示
print("もとの特徴量数:", features.shape[1])
print("削減後の特徴量数:", features_pca.shape[1])
