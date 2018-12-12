# -*- coding: utf-8 -*-

# ライブラリをロード
import nltk
from nltk import pos_tag
from nltk import word_tokenize

# 最初の1回は下のコメント外してリソースをダウンロード
# nltk.download('averaged_perceptron_tagger')

# テキストを生成
text_data = "Chris loved outdoor running"

# 訓練済み品詞タグ付け器を適用
text_tagged = pos_tag(word_tokenize(text_data))

# 品詞を表示
text_tagged

##########

# 品詞を用いて単語を選択
[word for word, tag in text_tagged if tag in ['NN','NNS','NNP','NNPS'] ]

##########
# ライブラリをロード
from sklearn.preprocessing import MultiLabelBinarizer

# テキストを生成
tweets = ["I am eating a burrito for breakfast",
          "Political science is an amazing field",
          "San Francisco is an awesome city"]

# リストを作成
tagged_tweets = []

# ツイート中の単語にタグ付け
for tweet in tweets:
    tweet_tag = pos_tag(word_tokenize(tweet))
    tagged_tweets.append([tag for word, tag in tweet_tag])

# ワンホットエンコードを用いて、タグを特徴量に変換
one_hot_multi = MultiLabelBinarizer()
one_hot_multi.fit_transform(tagged_tweets)

##########

# 特徴量名を表示
one_hot_multi.classes_

##########

# ライブラリをロード
from nltk.corpus import brown
from nltk.tag import UnigramTagger
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger

# 最初の1回は下のコメント外してリソースをダウンロード
# nltk.download('brown')


# Brown Corpusからテキストを取得して文章に分割
sentences = brown.tagged_sents(categories='news')

# 4000文を訓練データに、残り623文をテストデータに
train = sentences[:4000]
test = sentences[4000:]

# バックオフ付きタグ付け器を作成
unigram = UnigramTagger(train)
bigram = BigramTagger(train, backoff=unigram)
trigram = TrigramTagger(train, backoff=bigram)

# 精度を表示
trigram.evaluate(test)

