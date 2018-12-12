# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# URLを作成
url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/data.xlsx'

# データをロード
dataframe = pd.read_excel(url, sheetname=0, header=1)

# 最初の2行を表示
dataframe.head(2)
