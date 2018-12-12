# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# テキストを生成
text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])


# BoW特徴量行列を作成
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

# 特徴量行列を表示
bag_of_words

##########

bag_of_words.toarray()

##########

# 特徴量名を表示
count.get_feature_names()

##########

# パラメータを指定して特徴量行列を作成
count_2gram = CountVectorizer(ngram_range=(1,2),
                              stop_words="english",
                              vocabulary=['brazil'])
bag = count_2gram.fit_transform(text_data)

# 特徴量行列を表示
bag.toarray()

##########

# 1-gramと2-gramを表示
count_2gram.vocabulary_


