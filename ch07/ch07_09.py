# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd
import numpy as np

# 日時データを作成
time_index = pd.date_range("01/01/2010", periods=5, freq="M")

# データフレームを作成し、インデックスを設定
dataframe = pd.DataFrame(index=time_index)

# 欠損値を含む特徴量を作成
dataframe["Sales"] = [1.0,2.0,np.nan,np.nan,5.0]

# 欠損値を内挿して補完
dataframe.interpolate()

##########

# 前方補完
dataframe.ffill()

##########

# 後方補完
dataframe.bfill()

##########

# 欠損値を内挿
dataframe.interpolate(method="quadratic")

##########

# 欠損値を内挿
dataframe.interpolate(limit=1, limit_direction="forward")
