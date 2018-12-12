# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets

# データをロードして特徴量を2つだけ残す
boston = datasets.load_boston()
features = boston.data[:,0:2]
target = boston.target

# ランダムフォレスト回帰器を作成
randomforest = RandomForestRegressor(random_state=0, n_jobs=-1)

# ランダムフォレスト回帰器を訓練
model = randomforest.fit(features, target)
