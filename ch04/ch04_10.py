# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np

# 特徴量行列を作成
features = np.array([[1.1, 11.1],
                     [2.2, 22.2],
                     [3.3, 33.3],
                     [4.4, 44.4],
                     [np.nan, 55]])

# 欠損値のない(~ で条件を反転している)観測値だけを残す
features[~np.isnan(features).any(axis=1)]

##########

# ライブラリをロード
import pandas as pd

# データをロード
dataframe = pd.DataFrame(features, columns=["feature_1", "feature_2"])

# 欠損値のある観測値を削除
dataframe.dropna()
