# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn.preprocessing import Normalizer

# 特徴量行列を作成
features = np.array([[0.5, 0.5],
                     [1.1, 3.4],
                     [1.5, 20.2],
                     [1.63, 34.4],
                     [10.9, 3.3]])

# 正規化器を作成
normalizer = Normalizer(norm="l2")

# 特徴量行列を変換
normalizer.transform(features)

##########

# 特徴量行列を変換
features_l2_norm = Normalizer(norm="l2").transform(features)

# 特徴量行列を表示
features_l2_norm

##########

# 特徴量行列を変換
features_l1_norm = Normalizer(norm="l1").transform(features)

# 特徴量行列を表示
features_l1_norm

##########

# 総計を表示
print("最初の観測値の値の総計:",
      features_l1_norm[0, 0] + features_l1_norm[0, 1])

