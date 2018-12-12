# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# URLを作成
url = 'https://tinyurl.com/titanic-csv'

# データをロード
dataframe = pd.read_csv(url)

# 'sex'列が'female'の行のうち、最初の2行を表示
dataframe[dataframe['Sex'] == 'female'].head(2)

##########

# 行をフィルタリング
dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] >= 65)]

