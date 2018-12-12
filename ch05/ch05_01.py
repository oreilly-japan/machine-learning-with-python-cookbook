# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

# 特徴量を作成
feature = np.array([["Texas"],
                    ["California"],
                    ["Texas"],
                    ["Delaware"],
                    ["Texas"]])

# ワンホットエンコーダを作成
one_hot = LabelBinarizer()

# 特徴量をワンホットエンコード
one_hot.fit_transform(feature)

##########

# 特徴量クラスを表示
one_hot.classes_

##########

# ワンホットエンコードされた特徴量を逆変換
one_hot.inverse_transform(one_hot.transform(feature))

##########

# ライブラリをロード
import pandas as pd

# 特徴量からダミー変数を生成
pd.get_dummies(feature[:,0])

##########

# 複数クラス特徴量を作成
multiclass_feature = [("Texas", "Florida"),
                      ("California", "Alabama"),
                      ("Texas", "Florida"),
                      ("Delware", "Florida"),
                      ("Texas", "Alabama")]

# 複数クラス用ワンホットエンコーダを作成
one_hot_multiclass = MultiLabelBinarizer()

# 複数クラス特徴量をワンホットエンコード
one_hot_multiclass.fit_transform(multiclass_feature)

##########

# クラスを表示
one_hot_multiclass.classes_
