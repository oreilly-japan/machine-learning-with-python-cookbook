# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# データフレームを作成
dataframe = pd.DataFrame()

# 150の日時データを作る
dataframe['date'] = pd.date_range('1/1/2001', periods=150, freq='W')

# 年、月、日、時、分を特徴量として作成
dataframe['year'] = dataframe['date'].dt.year
dataframe['month'] = dataframe['date'].dt.month
dataframe['day'] = dataframe['date'].dt.day
dataframe['hour'] = dataframe['date'].dt.hour
dataframe['minute'] = dataframe['date'].dt.minute

# 3行表示
dataframe.head(3)
