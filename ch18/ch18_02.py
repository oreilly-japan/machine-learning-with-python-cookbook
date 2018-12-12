# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# テキストを生成
text_data = np.array(['I love Brazil. Brazil!',
                      'Brazil is best',
                      'Germany beats both'])

# BoW特徴量を作成
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

# 特徴量行列を作成
features = bag_of_words.toarray()

# ターゲットベクトルを作成
target = np.array([0,0,1])

# 事前確率を設定して、多項ナイーブベイズクラス分類器を作成
classifer = MultinomialNB(class_prior=[0.25, 0.5])

# 多項ナイーブベイズクラス分類器を訓練
model = classifer.fit(features, target)

###########

# 新たな観測値を作成
new_observation = [[0, 0, 0, 1, 0, 1, 0]]

# 新たな観測値のクラスを予測
model.predict(new_observation)


