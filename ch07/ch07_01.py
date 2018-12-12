# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
import pandas as pd

# 文字列を作成
date_strings = np.array(['03-04-2005 11:35 PM',
                         '23-05-2010 12:01 AM',
                         '04-09-2009 09:09 PM'])

# 日時データに変換
[pd.to_datetime(date, format='%d-%m-%Y %I:%M %p') for date in date_strings]

##########

# 日時データに変換
[pd.to_datetime(date, format="%d-%m-%Y %I:%M %p", errors="coerce")
for date in date_strings]


