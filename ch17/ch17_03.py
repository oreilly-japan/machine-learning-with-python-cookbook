# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 特徴量を標準化
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# サポートベクタクラス分類器を作成
svc = SVC(kernel="linear", probability=True, random_state=0)

# クラス分類器を訓練
model = svc.fit(features_standardized, target)

# 新たな観測値を作成
new_observation = [[.4, .4, .4, .4]]

# 予測確率を表示
model.predict_proba(new_observation)
