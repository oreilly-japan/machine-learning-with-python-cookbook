# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np

# 行列を作成
matrix_a = np.array([[1, 1],
                     [1, 2]])

# 行列を作成
matrix_b = np.array([[1, 3],
                     [1, 2]])

# 2つの行列を乗算
np.dot(matrix_a, matrix_b)

##########

# 2つの行列を乗算
matrix_a @ matrix_b

##########

# 2つの行列の要素ごとの乗算
matrix_a * matrix_b

