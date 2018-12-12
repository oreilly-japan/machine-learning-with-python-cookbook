# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 決定木クラス分類器オブジェクトを作成
decisiontree = DecisionTreeClassifier(random_state=0)

# 決定木クラス分類器を訓練
model = decisiontree.fit(features, target)

##########

# 新しい観測値を作成
observation = [[ 5,  4,  3,  2]]

# 観測値のクラスを予測
model.predict(observation)

##########

# 3つのクラスに対する予測クラス確率を表示
model.predict_proba(observation)

##########

# エントロピを用いる決定木クラス分類器オブジェクトを作成
decisiontree_entropy = DecisionTreeClassifier(
    criterion='entropy', random_state=0)

# 決定木クラス分類器を訓練
model_entropy = decisiontree_entropy.fit(features, target)


