# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np

# 行列を作成
matrix = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 8, 9]])

# 行列の行列式を算出
np.linalg.det(matrix)
