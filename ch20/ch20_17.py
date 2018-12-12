# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import models
from keras import layers

# 乱数シードを設定
np.random.seed(0)

# 利用したい特徴量の数を指定
number_of_features = 1000

# データをロード and target vector from movie review data
(data_train, target_train), (data_test, target_test) = imdb.load_data(
    num_words=number_of_features)

# Use padding or truncation to make each observation have 400 features
features_train = sequence.pad_sequences(data_train, maxlen=400)
features_test = sequence.pad_sequences(data_test, maxlen=400)

# ニューラルネットワークの作成を開始
network = models.Sequential()

# 埋め込み層を追加
network.add(layers.Embedding(input_dim=number_of_features, output_dim=128))

# 128ユニットのLSTM層を追加
network.add(layers.LSTM(units=128))

# 活性化関数としてシグモイド関数を用いる全結合層を追加
network.add(layers.Dense(units=1, activation="sigmoid"))

# ニューラルネットワークをコンパイル
network.compile(loss="binary_crossentropy", # クロスエントロピ
                optimizer="Adam", # Adam optimization
                metrics=["accuracy"]) # 性能指標は精度

# ニューラルネットワークを訓練
history = network.fit(features_train, # 特徴量
                      target_train, # ターゲット
                      epochs=3, # エポック数
                      verbose=0, # エポック毎の出力を抑制
                      batch_size=1000, # 1バッチあたりの観測値数
                      validation_data=(features_test, target_test)) # テストデータ

##########

# 最初の観測値を表示
print(data_train[0])

##########

# 最初の観測値を表示
print(features_test[0])
