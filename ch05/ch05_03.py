# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.feature_extraction import DictVectorizer

# 辞書を作成
data_dict = [{"Red": 2, "Blue": 4},
             {"Red": 4, "Blue": 3},
             {"Red": 1, "Yellow": 2},
             {"Red": 2, "Yellow": 2}]

# 辞書ベクトル変換器を作成
dictvectorizer = DictVectorizer(sparse=False)

# 辞書を特徴量行列に変換
features = dictvectorizer.fit_transform(data_dict)

# 特徴量行列を表示
features

##########

# 特徴量の名前を取得
feature_names = dictvectorizer.get_feature_names()

# 特徴量の名前を表示
feature_names

##########

# ライブラリをロード
import pandas as pd

# 特徴量からDataFrameを作成
pd.DataFrame(features, columns=feature_names)

##########

# 4つの文書に対する単語カウント辞書を作成
doc_1_word_count = {"Red": 2, "Blue": 4}
doc_2_word_count = {"Red": 4, "Blue": 3}
doc_3_word_count = {"Red": 1, "Yellow": 2}
doc_4_word_count = {"Red": 2, "Yellow": 2}

# リストを作成
doc_word_counts = [doc_1_word_count,
                   doc_2_word_count,
                   doc_3_word_count,
                   doc_4_word_count]

# 単語カウント辞書のリストを特徴量行列に変換
dictvectorizer.fit_transform(doc_word_counts)

