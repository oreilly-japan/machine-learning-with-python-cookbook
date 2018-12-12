
# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np

# 行列を作成
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 平均値を算出
np.mean(matrix)

# 分散を算出
np.var(matrix)

# 標準偏差を算出
np.std(matrix)

###########

# 各列の平均値を算出
np.mean(matrix, axis=0)
