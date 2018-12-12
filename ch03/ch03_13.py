# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# URLを作成
url = 'https://tinyurl.com/titanic-csv'

# データをロード
dataframe = pd.read_csv(url)

# 'Sex'列の値で行をグループ分けし、グループごとの平均値を計算
dataframe.groupby('Sex').mean()

##########

# 行をグループ分け
dataframe.groupby('Sex')

##########

# 行をグループ分けし、行数をカウント
dataframe.groupby('Survived')['Name'].count()

##########

# 行をグループ分けし、平均値を計算
dataframe.groupby(['Sex','Survived'])['Age'].mean()
