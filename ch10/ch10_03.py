# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd
import numpy as np

# 2つの特徴量が強く相関した特徴量行列を作成
features = np.array([[1, 1, 1],
                     [2, 2, 0],
                     [3, 3, 1],
                     [4, 4, 0],
                     [5, 5, 1],
                     [6, 6, 0],
                     [7, 7, 1],
                     [8, 7, 0],
                     [9, 7, 1]])

# 特徴量行列をDataFrameに変換
dataframe = pd.DataFrame(features)

# 相関行列を作成
corr_matrix = dataframe.corr().abs()

# 相関行列の上三角を選択
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                          k=1).astype(np.bool))

# 相関が0.95以上になる特徴量列のインデックスを抽出
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# 特徴量を削除
dataframe.drop(dataframe.columns[to_drop], axis=1).head(3)

##########

# 相関行列を表示
dataframe.corr()

##########

# 相関行列の上三角行列を表示
upper
