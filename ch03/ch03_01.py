# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# DataFrameを作成
# Create DataFrame
dataframe = pd.DataFrame()

# 列を追加
dataframe['Name'] = ['Jacky Jackson', 'Steven Stevenson']
dataframe['Age'] = [38, 25]
dataframe['Driver'] = [True, False]

# データフレームを表示
dataframe

##########

# 行を作成
new_person = pd.Series(['Molly Mooney', 40, True], index=['Name','Age','Driver'])

# 行を追加
dataframe.append(new_person, ignore_index=True)


