# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# ランダムフォレストクラス分類器を作成
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)

# ランダムフォレストクラス分類器を訓練
model = randomforest.fit(features, target)

# 新しい観測値を作成
observation = [[ 5,  4,  3,  2]]

# 観測値のクラスを予測
model.predict(observation)

##########

# 新しい観測値を作成
observation = [[ 5,  4,  3,  2]]

# 観測値のクラスを予測
model.predict(observation)

##########

# エントロピを用いるランダムフォレストクラス分類器オブジェクトを作成
randomforest_entropy = RandomForestClassifier(
    criterion="entropy", random_state=0)

# クラス分類器を訓練
model_entropy = randomforest_entropy.fit(features, target)
