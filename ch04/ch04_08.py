# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn.preprocessing import Binarizer

# 特徴量を作成
age = np.array([[6],
                [12],
                [20],
                [36],
                [65]])

# 二値化器を作成
# Create binarizer
binarizer = Binarizer(18)

# 特徴量を変換
binarizer.fit_transform(age)

##########

# 特徴量を複数のビンに分割
np.digitize(age, bins=[20,30,64])

##########

# 特徴量を複数のビンに分割
np.digitize(age, bins=[20,30,64], right=True)

##########

# 特徴量を複数のビンに分割
np.digitize(age, bins=[18])
