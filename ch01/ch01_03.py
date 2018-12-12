# -*- coding: utf-8 -*-

#ライブラリをロード
import numpy as np
from scipy import sparse

# 行列を作成
matrix = np.array([[0, 0],
                   [0, 1],
                   [3, 0]])

#  CSR(compressed sparse row) 形式の行列を作成
matrix_sparse = sparse.csr_matrix(matrix)

# 疎行列を表示
print(matrix_sparse)

#################

# 大きな行列を作成
# Create larger matrix
matrix_large = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

#  CSR(compressed sparse row) 形式の行列を作成
matrix_large_sparse = sparse.csr_matrix(matrix_large)

#################

# もとの疎行列を表示
print(matrix_sparse)

#################

# 大きな疎行列を表示
print(matrix_large_sparse)



