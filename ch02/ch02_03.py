# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# URLを作成
url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/data.csv'

# データセットをロード
dataframe = pd.read_csv(url)

# 最初の2行を表示
dataframe.head(2)
