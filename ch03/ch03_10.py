# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# URLを作成
url = 'https://tinyurl.com/titanic-csv'

# データをロード
dataframe = pd.read_csv(url)

# 列を削除
dataframe.drop('Age', axis=1).head(2)

##########

# 複数の列を削除
dataframe.drop(['Age', 'Sex'], axis=1).head(2)

##########

# 列を削除
dataframe.drop(dataframe.columns[1], axis=1).head(2)

##########

# 削除して新しいDataFrameを作成
dataframe_name_dropped = dataframe.drop(dataframe.columns[0], axis=1)

