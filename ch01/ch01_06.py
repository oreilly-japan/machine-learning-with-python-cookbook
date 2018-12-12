# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np

# 行列を作成
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 引数に100を加える関数を作成
add_100 = lambda i: i + 100

# ベクトル化された関数を作成
vectorized_add_100 = np.vectorize(add_100)

# この関数をmatrixのすべての要素に適用
vectorized_add_100(matrix)

###########

# すべての要素に100を加える
matrix + 100


