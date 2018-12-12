# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn.preprocessing import FunctionTransformer

# 特徴量行列を作成
features = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])

# 簡単な関数を定義
def add_ten(x):
    return x + 10

# 変換器を作成
ten_transformer = FunctionTransformer(add_ten)

# 特徴量行列を変換
ten_transformer.transform(features)

##########

# ライブラリをロード
import pandas as pd

# DataFrameを作成
df = pd.DataFrame(features, columns=["feature_1", "feature_2"])

# 関数を適用
df.apply(add_ten)
