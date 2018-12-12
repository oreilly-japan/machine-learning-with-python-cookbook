# -*- coding: utf-8 -*-

# ライブラリをロード
from nltk.stem.porter import PorterStemmer

# 単語トークンを作成
tokenized_words = ['i', 'am', 'humbled', 'by', 'this', 'traditional', 'meeting']

# 語幹抽出器を作成
porter = PorterStemmer()

# 語幹抽出器を適用
[porter.stem(word) for word in tokenized_words]
