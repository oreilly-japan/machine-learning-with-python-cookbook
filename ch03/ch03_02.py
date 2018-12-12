# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# URLを作成
url = 'https://tinyurl.com/titanic-csv'

# データをロード
dataframe = pd.read_csv(url)

# 最初の2行を表示
# Show two rows
dataframe.head(2)

##########

# データの形状を表示
dataframe.shape

##########

# 統計量を表示
dataframe.describe()


