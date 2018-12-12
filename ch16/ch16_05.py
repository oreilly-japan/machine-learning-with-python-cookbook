# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Make class highly imbalanced by removing first 40 observations
features = features[40:,:]
target = target[40:]

# Create target vector indicating if class 0, otherwise 1
target = np.where((target == 0), 0, 1)

# 特徴量を標準化
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# ロジスティック回帰器を作成 
logistic_regression = LogisticRegression(random_state=0, class_weight="balanced")

# ロジスティック回帰器を訓練
model = logistic_regression.fit(features_standardized, target)
