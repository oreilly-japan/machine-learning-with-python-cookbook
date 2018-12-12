# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# ランダムフォレストクラス分類器を作成
randomforest = RandomForestClassifier(
    random_state=0, n_estimators=1000, oob_score=True, n_jobs=-1)

# ランダムフォレストクラス分類器を訓練
model = randomforest.fit(features, target)

# OOBエラーを表示
randomforest.oob_score_
