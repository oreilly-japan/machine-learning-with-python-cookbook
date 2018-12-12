# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras import regularizers

# 乱数シードを設定
np.random.seed(0)

# 利用したい特徴量の数を指定
number_of_features = 1000

# 映画批評のデータとターゲットベクトルをロード
(data_train, target_train), (data_test, target_test) = imdb.load_data(
    num_words=number_of_features)

# 映画批評データをワンホットエンコードで特徴量行列に変換
tokenizer = Tokenizer(num_words=number_of_features)
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

# ニューラルネットワークの作成を開始
network = models.Sequential()

# 活性化関数としてReLUを用いる全結合層を追加
network.add(layers.Dense(units=16,
                         activation="relu",
                         kernel_regularizer=regularizers.l2(0.01),
                         input_shape=(number_of_features,)))

# 活性化関数としてReLUを用いる全結合層を追加
network.add(layers.Dense(units=16,
                         kernel_regularizer=regularizers.l2(0.01),
                         activation="relu"))

# 活性化関数としてシグモイド関数を用いる全結合層を追加
network.add(layers.Dense(units=1, activation="sigmoid"))

# ニューラルネットワークをコンパイル
network.compile(loss="binary_crossentropy", # クロスエントロピ
                optimizer="rmsprop", # 二乗平均平方根伝搬法
                metrics=["accuracy"]) # 性能指標は精度

# ニューラルネットワークを訓練
history = network.fit(features_train, # 特徴量
                      target_train, # ターゲットベクトル
                      epochs=3, # エポック数
                      verbose=0, # 出力しない
                      batch_size=100, # 1バッチあたりの観測値数
                      validation_data=(features_test, target_test)) # テストデータ

