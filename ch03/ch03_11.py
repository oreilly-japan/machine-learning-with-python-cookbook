# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# URLを作成
url = 'https://tinyurl.com/titanic-csv'

# データをロード
dataframe = pd.read_csv(url)

# 行を削除して最初の2行を表示
dataframe[dataframe['Sex'] != 'male'].head(2)

##########

# 行を削除して最初の2行を表示
dataframe[dataframe['Name'] != 'Allison, Miss Helen Loraine'].head(2)

##########

# 行を削除して最初の2行を表示
dataframe[dataframe.index != 0].head(2)
