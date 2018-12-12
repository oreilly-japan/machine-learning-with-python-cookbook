# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures

# データをロードして特徴量を一つだけ残す
boston = load_boston()
features = boston.data[:,0:1]
target = boston.target

# 多項式特徴量x^2とx^3を作成
polynomial = PolynomialFeatures(degree=3, include_bias=False)
features_polynomial = polynomial.fit_transform(features)

# 線形回帰器を作成
regression = LinearRegression()

# 線形回帰器を訓練
model = regression.fit(features_polynomial, target)

##########

# 最初の観測値を表示
features[0]

##########

# 最初の観測値を二乗して得たx^2を表示
features[0]**2

##########

# 最初の観測値を三乗して得たx^3を表示
features[0]**3

##########

# 最初の観測値のx、x^2、x^3の値を表示
features_polynomial[0]

