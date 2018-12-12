# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd
from sqlalchemy import create_engine

# データベース接続を作成
database_connection = create_engine('sqlite:///sample.db')

# データをロード
dataframe = pd.read_sql_query('SELECT * FROM data', database_connection)

# 最初の2行を表示
dataframe.head(2)
