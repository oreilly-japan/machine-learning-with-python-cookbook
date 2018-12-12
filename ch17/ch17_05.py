# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

#データをロードして2クラスだけ残す
iris = datasets.load_iris()
features = iris.data[:100,:]
target = iris.target[:100]

# 観測値の最初の40個を捨ててクラスのバランスの崩す
features = features[40:,:]
target = target[40:]

# クラス0かどうかを示すターゲットベクタを作成
target = np.where((target == 0), 0, 1)

# 特徴量を標準化
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# サポートベクタクラス分類器を作成
svc = SVC(kernel="linear", class_weight="balanced", C=1.0, random_state=0)

# サポートベクタクラス分類器を訓練
model = svc.fit(features_standardized, target)
