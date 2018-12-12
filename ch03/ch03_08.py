# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# URLを作成
url = 'https://tinyurl.com/titanic-csv'

# データをロード
dataframe = pd.read_csv(url)

# ユニークな値のリストを取得
dataframe['Sex'].unique()

##########

# 現れた回数を表示
dataframe['Sex'].value_counts()

##########

# カウント数を表示
dataframe['PClass'].value_counts()

##########

# ユニークな値の数を表示
dataframe['PClass'].nunique()
