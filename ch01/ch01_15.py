# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np

# 行列を作成
matrix = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 8, 9]])

# トレースを返す
matrix.trace()

##########

# 対角要素を足し合わせたものを返す
sum(matrix.diagonal())
