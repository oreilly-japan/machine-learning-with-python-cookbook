# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np

# 行列を作成
matrix = np.array([[1, -1, 3],
                   [1, 1, 6],
                   [3, 8, 9]])

# 固有値と固有ベクトルを計算
eigenvalues, eigenvectors = np.linalg.eig(matrix)

# 固有値を表示
eigenvalues

# 固有ベクトルを表示
eigenvectors


