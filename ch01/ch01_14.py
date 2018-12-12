# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np

# 行列を作成
matrix = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 8, 9]])

# 対角要素を返す
matrix.diagonal()

##########

# 主対角要素の一つ上の副対角要素を返す
matrix.diagonal(offset=1)

##########

# 主対角要素の一つ下の副対角要素を返す
matrix.diagonal(offset=-1)
