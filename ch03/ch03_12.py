# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# URLを作成
url = 'https://tinyurl.com/titanic-csv'

# データを作成
dataframe = pd.read_csv(url)

# 重複した行を削除し、最初の2行を出力
dataframe.drop_duplicates().head(2)

##########

# 行数を表示
print("もとのDataFrame中の行数:", len(dataframe))
print("重複削除後の行数:", len(dataframe.drop_duplicates()))

##########

# 重複を削除
dataframe.drop_duplicates(subset=['Sex'])

##########

# 重複を削除
dataframe.drop_duplicates(subset=['Sex'], keep='last')

