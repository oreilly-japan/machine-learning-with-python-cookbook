# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn import linear_model, datasets

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# ロジスティック回帰交差検証器を作成
logit = linear_model.LogisticRegressionCV(Cs=100)

# 交差検証器を訓練
logit.fit(features, target)

###########
