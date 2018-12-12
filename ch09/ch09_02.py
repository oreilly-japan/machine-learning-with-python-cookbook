# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles

# 線形分離不可能なデータを生成
features, _ = make_circles(n_samples=1000, random_state=1, noise=0.1, factor=0.1)

# RBF(radius basis function)カーネルPCAを適用
kpca = KernelPCA(kernel="rbf", gamma=15, n_components=1)
features_kpca = kpca.fit_transform(features)

print("もとの特徴量数:", features.shape[1])
print("削減後の特徴量数:", features_kpca.shape[1])
