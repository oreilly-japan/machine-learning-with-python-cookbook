# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# 日時データを作成
time_index = pd.date_range("01/01/2010", periods=5, freq="M")

# データフレームを作成し、インデックスを設定
dataframe = pd.DataFrame(index=time_index)

# 特徴量を作成
dataframe["Stock_Price"] = [1,2,3,4,5]

# 移動平均を計算
dataframe.rolling(window=2).mean()

