# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# adaboost決定木クラス分類器を作成
adaboost = AdaBoostClassifier(random_state=0)

# adaboost決定木クラス分類器を訓練
model = adaboost.fit(features, target)

