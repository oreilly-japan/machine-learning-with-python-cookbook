# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# URLを作成
url = 'https://tinyurl.com/titanic-csv'

# データをロード
dataframe = pd.read_csv(url)

# 関数を定義
def uppercase(x):
    return x.upper()

# 関数を適用して、結果の最初の2行を表示
dataframe['Name'].apply(uppercase)[0:2]



