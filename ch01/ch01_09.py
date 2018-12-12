# -*- coding: utf-8 -*-
# ライブラリをロード
import numpy as np

# 4x3の行列を作成
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

# 2x6の行列に形状変更
matrix.reshape(2, 6)

matrix.size

##########

matrix.reshape(1, -1)

##########

matrix.reshape(12)
