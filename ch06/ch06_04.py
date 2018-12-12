# -*- coding: utf-8 -*-

# ライブラリをロード
from nltk.tokenize import word_tokenize

# 最初の1回は下のコメント外してリソースをダウンロード
# import nltk
# nltk.download('punkt')

# テキストを生成
string = "The science of today is the technology of tomorrow"

# 単語単位でトークン化
word_tokenize(string)

##########

# ライブラリをロード
from nltk.tokenize import sent_tokenize

# テキストを生成
string = "The science of today is the technology of tomorrow. Tomorrow is today."

# 文章単位でトークン化
sent_tokenize(string)

