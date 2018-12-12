# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 最初の40観測値を削除して、クラスを不均等に
features = features[40:,:]
target = target[40:]

# クラスが0なら0、それ以外なら1となるターゲットベクタを作成
# target = np.where((target == 0), 0, 1)

# ランダムフォレストクラス分類器を作成
randomforest = RandomForestClassifier(
    random_state=0, n_jobs=-1, class_weight="balanced")

# ランダムフォレストクラス分類器を訓練
model = randomforest.fit(features, target)

##########

# 観測値数の少ないクラスの重みを計算
110/(2*10)

##########

# 観測値数の多いクラスの重みを計算
110/(2*100)


