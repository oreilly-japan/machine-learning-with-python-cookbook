# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 決定木クラス分類器を作成
decisiontree = DecisionTreeClassifier(random_state=0,
                                      max_depth=None,
                                      min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0,
                                      max_leaf_nodes=None,
                                      min_impurity_decrease=0)

# 決定木クラス分類器を訓練
model = decisiontree.fit(features, target)
