# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np

# 行列を作成
matrix_a = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 2]])

# 行列を作成
matrix_b = np.array([[1, 3, 1],
                     [1, 3, 1],
                     [1, 3, 8]])

# 2つの行列を加算
np.add(matrix_a, matrix_b)

##########

# 2つの行列の減算
np.subtract(matrix_a, matrix_b)

##########

# 2つの行列の加算
matrix_a + matrix_b
