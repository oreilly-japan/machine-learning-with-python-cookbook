# -*- coding: utf-8 -*-


# ライブラリをロード
from sklearn.feature_selection import VarianceThreshold

# 特徴量行列を下記のように作成:
# 特徴量 0: 80% クラス 0
# 特徴量 1: 80% クラス 1
# 特徴量 2: 60% クラス 0, 40% クラス 1
features = [[0, 1, 0],
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0]]

# 分散で閾値処理
thresholder = VarianceThreshold(threshold=(.75 * (1 - .75)))
thresholder.fit_transform(features)

