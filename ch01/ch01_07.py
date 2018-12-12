# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np

# 行列を作成
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 最大の要素を返す
# Return maximum element
np.max(matrix)

##########

# 最小の要素を返す
np.min(matrix)

##########

# 各列における最大要素を見つける
np.max(matrix, axis=0)

##########

# 各行における最大要素を見つける
np.max(matrix, axis=1)

