# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# 特徴量を作成
dataframe = pd.DataFrame({"Score": ["Low", "Low", "Medium", "Medium", "High"]})

# マップを作成
scale_mapper = {"Low":1,
                "Medium":2,
                "High":3}

# 特徴量の値をマップを使って置換
dataframe["Score"].replace(scale_mapper)

##########

dataframe = pd.DataFrame({"Score": ["Low",
                                    "Low",
                                    "Medium",
                                    "Medium",
                                    "High",
                                    "Barely More Than Medium"]})

scale_mapper = {"Low":1,
                "Medium":2,
                "Barely More Than Medium": 3,
                "High":4}

dataframe["Score"].replace(scale_mapper)

##########

scale_mapper = {"Low":1,
                "Medium":2,
                "Barely More Than Medium": 2.1,
                "High":3}

dataframe["Score"].replace(scale_mapper)
