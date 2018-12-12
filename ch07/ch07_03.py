# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# データフレームを作成
dataframe = pd.DataFrame()

# 日時データを作成
dataframe['date'] = pd.date_range('1/1/2001', periods=100000, freq='H')

# 2つの日時の間の観測値を選択
dataframe[(dataframe['date'] > '2002-1-1 01:00:00') &
          (dataframe['date'] <= '2002-1-1 04:00:00')]

##########

# インデックスとして指定
dataframe = dataframe.set_index(dataframe['date'])

# 2つの日時の間の観測値を選択
dataframe.loc['2002-1-1 01:00:00':'2002-1-1 04:00:00']
