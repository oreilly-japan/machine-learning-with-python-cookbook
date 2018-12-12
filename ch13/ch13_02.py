# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures

# データをロード with only two features
boston = load_boston()
features = boston.data[:,0:2]
target = boston.target

# 交互作用の項を作成
interaction = PolynomialFeatures(
    degree=3, include_bias=False, interaction_only=True)
features_interaction = interaction.fit_transform(features)

# 線形回帰器を作成
regression = LinearRegression()

# 線形回帰器を訓練
model = regression.fit(features_interaction, target)

##########

# 最初の観測値の特徴量の値を表示
features[0]

##########

# ライブラリをロード
import numpy as np

# 個々の観測値に対して、最初の特徴量値と2つ目の特徴量値を掛け合わせる
interaction_term = np.multiply(features[:, 0], features[:, 1])

##########

# 最初の観測値の交互作用項を表示
interaction_term[0]

##########

# 最初の観測値の値を表示
features_interaction[0]




