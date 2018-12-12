# -*- coding: utf-8 -*-
# ライブラリをロード
import pandas as pd

# URLを作成
url = 'https://tinyurl.com/titanic-csv'

# データをロード
dataframe = pd.read_csv(url)

# 最初の行を選択
dataframe.iloc[0]

############

# 3行を選択
dataframe.iloc[1:4]

############

# 4行を選択
dataframe.iloc[:4]

############

# インデックスを設定
dataframe = dataframe.set_index(dataframe['Name'])

# 行を表示
dataframe.loc['Allen, Miss Elisabeth Walton']


