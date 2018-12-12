
# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.datasets import make_regression

# 特徴量行列、ターゲットベクトル、生成に用いた係数の真の値を生成
features, target, coefficients = make_regression(n_samples = 100,
                                                 n_features = 3,
                                                 n_informative = 3,
                                                 n_targets = 1,
                                                 noise = 0.0,
                                                 coef = True,
                                                 random_state = 1)

# 特徴量行列とターゲットベクトルを表示
print('特徴量行列\n', features[:3])
print('ターゲットベクトル\n', target[:3])

##########

# ライブラリをロード
from sklearn.datasets import make_classification

# 特徴量行列、ターゲットベクトルを生成
features, target = make_classification(n_samples = 100,
                                       n_features = 3,
                                       n_informative = 3,
                                       n_redundant = 0,
                                       n_classes = 2,
                                       weights = [.25, .75],
                                       random_state = 1)

# 特徴量行列、ターゲットベクトルを表示
print('特徴量行列\n', features[:3])
print('ターゲットベクトル\n', target[:3])

##########

# ライブラリをロード
from sklearn.datasets import make_blobs

# 特徴量行列、ターゲットベクトルを生成
features, target = make_blobs(n_samples = 100,
                              n_features = 2,
                              centers = 3,
                              cluster_std = 0.5,
                              shuffle = True,
                              random_state = 1)

# 特徴量行列、ターゲットベクトルを表示
print('特徴量行列\n', features[:3])
print('ターゲットベクトル\n', target[:3])

##########

# ライブラリをロード
import matplotlib.pyplot as plt

# 散布プロットを表示
plt.scatter(features[:,0], features[:,1], c=target)
plt.show()
