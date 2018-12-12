# -*- coding: utf-8 -*-

# ライブラリをロード
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer

# テキストを作成
texts = ["蔵王は、山形県と宮城県の県境にあります。",
         "山形県の月山は、7月までスキーができるそうです。",
         "宮城県の松島は日本三景の一つです。"]

# トークン化器を作成
t = Tokenizer()

# 文字列をトークン列に変換し、
# トークン列から不要な品詞のトークンを省き
# 基本形の列を返す関数を定義
def japaneseTokenize(text):
    tokens = t.tokenize(text)
    return [token.base_form for token in tokens
            if not token.part_of_speech.split(',')[0] in ['助詞','助動詞','記号','接頭詞']]

# BoW特徴量行列を作成
count = CountVectorizer(analyzer=japaneseTokenize)
bag_of_words = count.fit_transform(texts)

# 特徴量行列を表示
bag_of_words

##########

# 密行列として表示
bag_of_words.toarray()

##########

# 特徴量名を表示
count.get_feature_names()


##########

# ライブラリをロード
from sklearn.feature_extraction.text import TfidfVectorizer

# tf-idf特徴量行列を作成
tfidf = TfidfVectorizer(analyzer=japaneseTokenize)
feature_matrix = tfidf.fit_transform(texts)

# tf-idf特徴量行列を表示
feature_matrix.toarray()



