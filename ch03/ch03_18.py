# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# DataFrameを作成
data_a = {'id': ['1', '2', '3'],
          'first': ['Alex', 'Amy', 'Allen'],
          'last': ['Anderson', 'Ackerman', 'Ali']}
dataframe_a = pd.DataFrame(data_a, columns = ['id', 'first', 'last'])

# DataFrameを作成
data_b = {'id': ['4', '5', '6'],
          'first': ['Billy', 'Brian', 'Bran'],
          'last': ['Bonder', 'Black', 'Balwner']}
dataframe_b = pd.DataFrame(data_b, columns = ['id', 'first', 'last'])

# 行方向にDataFrameを連結
pd.concat([dataframe_a, dataframe_b], axis=0)

##########

# 列方向にDataFrameを連結
pd.concat([dataframe_a, dataframe_b], axis=1)

##########

# 行を作成
row = pd.Series([10, 'Chris', 'Chillon'], index=['id', 'first', 'last'])

# 行を追加
dataframe_a.append(row, ignore_index=True)

