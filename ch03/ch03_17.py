# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# URLを作成
url = 'https://tinyurl.com/titanic-csv'

# データをロード
dataframe = pd.read_csv(url)

# 行をグループ分けし、関数をグループごとに適用
dataframe.groupby('Sex').apply(lambda x: x.count())



