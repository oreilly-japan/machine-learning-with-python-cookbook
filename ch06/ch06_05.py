# -*- coding: utf-8 -*-

# ライブラリをロード
from nltk.corpus import stopwords

# 最初の一回は下のコメントを外して、ストップワードをダウンロード
# import nltk
# nltk.download('stopwords')

# 単語トークン列を作成
# Create word tokens
tokenized_words = ['i',
                   'am',
                   'going',
                   'to',
                   'go',
                   'to',
                   'the',
                   'store',
                   'and',
                   'park']

# ストップワードをロード
stop_words = stopwords.words('english')

# ストップワードを削除
[word for word in tokenized_words if word not in stop_words]

##########

# ストップワードを表示
stop_words[:5]


