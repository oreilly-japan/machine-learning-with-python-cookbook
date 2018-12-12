# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn import preprocessing

# 特徴量を作成
feature = np.array([[-500.5],
                    [-100.1],
                    [0],
                    [100.1],
                    [900.9]])

# スケール変換器を作成
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

# 特徴量をスケール変換
scaled_feature = minmax_scale.fit_transform(feature)

# 特徴量を表示
scaled_feature
