# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# データをロード
boston = load_boston()
features = boston.data
target = boston.target

# 特徴量を標準化
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# alpha を指定してリッジ回帰器を作成
regression = Ridge(alpha=0.5)

# リッジ回帰を訓練
model = regression.fit(features_standardized, target)

##########

# ライブラリをロード
from sklearn.linear_model import RidgeCV

# 3つのalpha値を指定してリッジ回帰器を作成
regr_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])

# リッジ回帰器を訓練
model_cv = regr_cv.fit(features_standardized, target)

# 係数を表示
model_cv.coef_

##########

# alphaを表示
model_cv.alpha_

