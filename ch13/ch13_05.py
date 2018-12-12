# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# データをロード
boston = load_boston()
features = boston.data
target = boston.target

# 特徴量を標準化
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# alphaを指定してLasso回帰器を作成
regression = Lasso(alpha=0.5)

# Lasso回帰器を訓練
model = regression.fit(features_standardized, target)

##########

# 係数を表示
model.coef_

##########

# alphaを大きくしてLasso回帰器を作成
regression_a10 = Lasso(alpha=10)
model_a10 = regression_a10.fit(features_standardized, target)
model_a10.coef_

