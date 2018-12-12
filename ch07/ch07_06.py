# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# 日時データを作成
dates = pd.Series(pd.date_range("2/2/2002", periods=3, freq="M"))

# 曜日を表示
dates.dt.weekday_name

##########

# 曜日を表示
dates.dt.weekday
