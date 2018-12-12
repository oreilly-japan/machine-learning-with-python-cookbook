# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# URLを作成
url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/data.json'

# データをロード
dataframe = pd.read_json(url, orient='columns')

# 最初の2行を表示
dataframe.head(2)

