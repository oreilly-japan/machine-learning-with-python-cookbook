# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets

# データをロードして、2つだけ特徴量を残す。
boston = datasets.load_boston()
features = boston.data[:,0:2]
target = boston.target

# 決定木回帰器オブジェクトを作成
decisiontree = DecisionTreeRegressor(random_state=0)

# 決定木回帰器を訓練
model = decisiontree.fit(features, target)

##########

# 新しい観測値の作成
observation = [[0.02, 16]]

# 観測値のターゲット値を予測
model.predict(observation)

##########

# MAE(平均絶対誤差)を用いる決定木回帰器オブジェクトを作成 
decisiontree_mae = DecisionTreeRegressor(criterion="mae", random_state=0)

# 決定木回帰器を訓練
model_mae = decisiontree_mae.fit(features, target)
