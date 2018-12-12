# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# データフレームを作成
dataframe = pd.DataFrame()

# 2つの日時特徴量を作成
dataframe['Arrived'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-04-2017')]
dataframe['Left'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-06-2017')]

# 2つの日時特徴量の差を計算
dataframe['Left'] - dataframe['Arrived']

##########

# 2つの日時特徴量の差を計算
pd.Series(delta.days for delta in (dataframe['Left'] - dataframe['Arrived']))
