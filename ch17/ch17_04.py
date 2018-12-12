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

# 特徴量を標準化
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# サポートベクタクラス分類器を作成
svc = SVC(kernel="linear", random_state=0)

# サポートベクタクラス分類器を訓練
model = svc.fit(features_standardized, target)

# サポートベクタを表示
model.support_vectors_

###########

model.support_

###########

model.n_support_

