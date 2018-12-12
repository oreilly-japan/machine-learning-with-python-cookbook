# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.datasets import load_boston
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split

# データをロード
boston = load_boston()

# 特徴量を作成
features, target = boston.data, boston.target

# 訓練セットとテストセットに分割
features_train, features_test, target_train, target_test = train_test_split(
    features, target, random_state=0)

# ダミー回帰器を作成
dummy = DummyRegressor(strategy='mean')
# ダミー回帰器を「訓練」
dummy.fit(features_train, target_train)

# R^2スコアを取得
dummy.score(features_test, target_test)

##########

# ライブラリをロード
from sklearn.linear_model import LinearRegression

# 単純な線形回帰モデルを訓練
ols = LinearRegression()
ols.fit(features_train, target_train)

# R^2スコアを取得
ols.score(features_test, target_test)

##########

# 常に20と予測するダミー回帰モデルを作成
clf = DummyRegressor(strategy='constant', constant=20)
clf.fit(features_train, target_train)

# スコアを評価
clf.score(features_test, target_test)


