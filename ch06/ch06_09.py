# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# テキストを生成
text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])

# tf-idf特徴量行列を作成
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)

# tf-idf特徴量行列を表示
feature_matrix

##########

# tf-idf特徴量行列を密行列として表示
feature_matrix.toarray()

##########

# 特徴量名を表示
tfidf.vocabulary_


