# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# データをロードし、特徴量を2つに制限
boston = load_boston()
features = boston.data[:,0:2]
target = boston.target

# 線形回帰器を作成
regression = LinearRegression()

# 線形回帰器を訓練
model = regression.fit(features, target)

##########

# 切片(intercept)を表示
model.intercept_

##########

# 特徴量の係数(coefficient)を表示
model.coef_

##########

# ターゲットベクトルの最初の値に1000をかける
target[0]*1000

##########

# 最初の観測値のターゲット値を予測して、1000をかける
model.predict(features)[0]*1000


