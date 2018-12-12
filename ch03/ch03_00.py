# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# URLの作成
url = 'https://tinyurl.com/titanic-csv'

# データをデータフレームとしてロード
dataframe = pd.read_csv(url)

# 最初の5行を表示
dataframe.head(5)
