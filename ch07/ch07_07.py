# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# データフレームを作成
dataframe = pd.DataFrame()

# 日付データを作成
dataframe["dates"] = pd.date_range("1/1/2001", periods=5, freq="D")
dataframe["stock_price"] = [1.1,2.2,3.3,4.4,5.5]


# 1行分ラグのある(遅れている)値を作成
dataframe["previous_days_stock_price"] = dataframe["stock_price"].shift(1)

# データフレームを表示
dataframe
