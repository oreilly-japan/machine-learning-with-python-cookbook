# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# 日時データを作成
pd.Timestamp('2017-05-01 06:00:00', tz='Europe/London')

##########

# 日時データを作成
date = pd.Timestamp('2017-05-01 06:00:00')

# タイムゾーンを設定
date_in_london = date.tz_localize('Europe/London')

# 日時データを表示
date_in_london

##########

# タイムゾーンを変更
date_in_london.tz_convert('Africa/Abidjan')

##########

# 日時データを3つ作成
dates = pd.Series(pd.date_range('2/2/2002', periods=3, freq='M'))

# タイムゾーンを設定
dates.dt.tz_localize('Africa/Abidjan')

##########

# ライブラリをロード
from pytz import all_timezones

# タイムゾーンを2つ表示
all_timezones[0:2]
