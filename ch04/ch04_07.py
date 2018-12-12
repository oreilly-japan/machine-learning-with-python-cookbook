# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# DataFrameを作成
houses = pd.DataFrame()
houses['Price'] = [534433, 392333, 293222, 4322032]
houses['Bathrooms'] = [2, 3.5, 2, 116]
houses['Square_Feet'] = [1500, 2500, 1500, 48000]

# 観測値をフィルタリング
houses[houses['Bathrooms'] < 20]

##########

# ライブラリをロード
import numpy as np

# 真偽条件に基づいて特徴量を作る
houses["Outlier"] = np.where(houses["Bathrooms"] < 20, 0, 1)

# データを表示
houses

##########

# 特徴量を対数にする
houses["Log_Of_Square_Feet"] = [np.log(x) for x in houses["Square_Feet"]]

# データを表示
houses
