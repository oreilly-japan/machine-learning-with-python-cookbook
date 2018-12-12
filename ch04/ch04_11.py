# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# 人工的な特徴量行列を作成
features, _ = make_blobs(n_samples = 1000,
                         n_features = 2,
                         random_state = 1)

# 特徴量を標準化
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

# 最初の特徴量の最初の値を欠損値に置換
true_value = standardized_features[0,0]
standardized_features[0,0] = np.nan

# 特徴量行列中の欠損値を補完
features_knn_imputed = KNN(k=5, verbose=0).complete(standardized_features)

# 真の値と補完された値を比較
print("真の値:", true_value)
print("補完された値:", features_knn_imputed[0,0])

##########

# ライブラリをロード
from sklearn.preprocessing import Imputer

# 欠損値補完器(imputer)を作る
# Create imputer
mean_imputer = Imputer(strategy="mean", axis=0)

# 欠損値を補完する
features_mean_imputed = mean_imputer.fit_transform(features)

# 真の値と補完された値を比較
print("真の値:", true_value)
print("補完された値:", features_mean_imputed[0,0])
