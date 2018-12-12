# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# URLを作成
url = 'https://tinyurl.com/titanic-csv'

# データをロード
dataframe = pd.read_csv(url)

# 列名を変更して, 最初の2行を表示
dataframe.rename(columns={'PClass': 'Passenger Class'}).head(2)

##########

# 列名を変更して, 最初の2行を表示
dataframe.rename(columns={'PClass': 'Passenger Class', 'Sex': 'Gender'}).head(2)

##########

# ライブラリをロード
import collections

# ディクショナリを作成
column_names = collections.defaultdict(str)

# キーを作成
for name in dataframe.columns:
    column_names[name]

# ディクショナリを表示
column_names
