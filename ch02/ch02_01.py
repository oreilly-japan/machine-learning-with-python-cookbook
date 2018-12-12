# -*- coding: utf-8 -*-

# scikit-learnのdatasetsをロード
from sklearn import datasets

# digitデータセットをロード
digits = datasets.load_digits()

# 特徴量行列を作成
features = digits.data

# ターゲットベクトルを作成
target = digits.target

# 最初の観測を表示
features[0]
