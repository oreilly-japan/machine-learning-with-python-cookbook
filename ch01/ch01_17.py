# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np

# 2つのベクトルを作成
vector_a = np.array([1,2,3])
vector_b = np.array([4,5,6])

# 内積を計算
np.dot(vector_a, vector_b)

########## 

# 内積を計算
vector_a @ vector_b
