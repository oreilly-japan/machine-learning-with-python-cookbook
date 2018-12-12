# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# URLを作成
url = 'https://tinyurl.com/titanic-csv'

# データをロード
dataframe = pd.read_csv(url)

# 最初の2つの名前を大文字にして表示
for name in dataframe['Name'][0:2]:
    print(name.upper())

##########

# 最初の2つの名前を大文字にして表示
[name.upper() for name in dataframe['Name'][0:2]]

