# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# URLを作成
url = 'https://tinyurl.com/titanic-csv'

# データをロード
dataframe = pd.read_csv(url)

# 値を置き換えて、最初の2行を表示
dataframe['Sex'].replace("female", "Woman").head(2)

##########

# "female"と"male"を"Woman"と"Man"にそれぞれ置き換える。
dataframe['Sex'].replace(["female", "male"], ["Woman", "Man"]).head(5)

##########

# 値を置き換えて、最初の2行を表示
dataframe.replace(1, "One").head(2)

##########

# 値を置き換えて、最初の2行を表示
dataframe.replace(r"1st", "First", regex=True).head(2)
