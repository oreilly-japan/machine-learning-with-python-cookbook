# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

# データをロード
iris = load_iris()

# ターゲットベクトルと特徴量行列を作成
features, target = iris.data, iris.target

# 訓練セットとテストセットに分割
features_train, features_test, target_train, target_test = train_test_split(
features, target, random_state=0)

# ダミークラス分類器を作成
dummy = DummyClassifier(strategy='uniform', random_state=1)

# モデルを「訓練」
dummy.fit(features_train, target_train)

# 精度スコアを計算
dummy.score(features_test, target_test)

##########

# ライブラリをロード
from sklearn.ensemble import RandomForestClassifier

# クラス分類器を作成
classifier = RandomForestClassifier()

# モデルを訓練
classifier.fit(features_train, target_train)

# 精度スコアを計算
classifier.score(features_test, target_test)
