
# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn import preprocessing

# 特徴量を作成
x = np.array([[-1000.1],
              [-200.2],
              [500.5],
              [600.6],
              [9000.9]])

# スケール変換器を作成
scaler = preprocessing.StandardScaler()

# 特徴量を変換
standardized = scaler.fit_transform(x)

# 特徴量を表示
standardized

##########

# 平均と標準偏差を表示
print("平均:", round(standardized.mean()))
print("標準偏差:", standardized.std())

##########

# スケール変換器を作成
robust_scaler = preprocessing.RobustScaler()

# 特徴量を変換
robust_scaler.fit_transform(x)
