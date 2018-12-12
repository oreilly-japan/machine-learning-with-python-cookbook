# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# URLを作成
url = 'https://tinyurl.com/titanic-csv'

# データをロード
dataframe = pd.read_csv(url)

## 欠損値を選択し、2つを表示
dataframe[dataframe['Age'].isnull()].head(2)

##########

# 値をNaNで置き換えることを試みる
dataframe['Sex'] = dataframe['Sex'].replace('male', NaN)

##########

# ライブラリをロード
import numpy as np

# 値をNaNで置き換える
dataframe['Sex'] = dataframe['Sex'].replace('male', np.nan)

##########

# 欠損値の表現を指定してデータをロード
dataframe = pd.read_csv(url, na_values=[np.nan, 'NONE', -999])
