# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np

# 行列を作成
matrix = np.array([[1, 4],
                   [2, 5]])

# 逆行列を算出
# Calculate inverse of matrix
np.linalg.inv(matrix)

##########

# 行列とその逆行列を乗算
matrix @ np.linalg.inv(matrix)
