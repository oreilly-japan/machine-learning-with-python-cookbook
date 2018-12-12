# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd
import numpy as np

# シード値を設定
np.random.seed(0)

# 日時の範囲を作成
time_index = pd.date_range('06/06/2017', periods=100000, freq='30S')

# DataFrameを作成
dataframe = pd.DataFrame(index=time_index)

# ランダムな値の行を作成
dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)

# 一週間ごとにグループ分けして、週ごとに値を集計

dataframe.resample('W').sum()

###########

# 3行表示
dataframe.head(3)

###########

# 2週間ごとにグループ分けして平均値を計算
dataframe.resample('2W').mean()

###########

# 月ごとにグループ分けして、行の数を数える
dataframe.resample('M').count()

###########

# 月ごとにグループ分けして、行の数を数える
dataframe.resample('M', label='left').count()
