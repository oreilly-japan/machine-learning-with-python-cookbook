# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.models import load_model

# 乱数シードを設定
np.random.seed(0)

# 利用したい特徴量の数を指定
number_of_features = 1000

# 映画批評データのデータとターゲットベクトルをロードする
(train_data, train_target), (test_data, test_target) = imdb.load_data(
    num_words=number_of_features)

# 映画批評データをワンホットエンコードして特徴量行列に変換
# Convert movie review data to a one-hot encoded feature matrix
tokenizer = Tokenizer(num_words=number_of_features)
train_features = tokenizer.sequences_to_matrix(train_data, mode="binary")
test_features = tokenizer.sequences_to_matrix(test_data, mode="binary")

# ニューラルネットワークの作成を開始
network = models.Sequential()

# 活性化関数としてReLUを用いる全結合層を追加
network.add(layers.Dense(units=16,
                         activation="relu",
                         input_shape=(number_of_features,)))

# 活性化関数としてシグモイド関数を用いる全結合層を追加
network.add(layers.Dense(units=1, activation="sigmoid"))

# ニューラルネットワークをコンパイル
network.compile(loss="binary_crossentropy", # クロスエントロピ
                optimizer="rmsprop", # 二乗平均平方根伝搬法
                metrics=["accuracy"]) # 性能指標は精度

# ニューラルネットワークを訓練
history = network.fit(train_features, # 特徴量
                      train_target, # ターゲットベクトル
                      epochs=3, # エポック数
                      verbose=0, # No output
                      batch_size=100, # 1バッチあたりの観測値数
                      validation_data=(test_features, test_target)) # テストデータ

# ニューラルネットワークをセーブ
network.save("model.h5")

###########

# ニューラルネットワークをロード
network = load_model("model.h5")
