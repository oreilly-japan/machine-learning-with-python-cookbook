# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers

# 乱数シードを設定
np.random.seed(0)

# 利用したい特徴量の数を指定
number_of_features = 5000

# 特徴量とターゲットベクトルをロード
data = reuters.load_data(num_words=number_of_features)
(data_train, target_vector_train), (data_test, target_vector_test) = data

# 特徴量をワンホットエンコードして特徴量行列に変換
tokenizer = Tokenizer(num_words=number_of_features)
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

# ターゲットベクトルをワンホットエンコードしてターゲット行列を作成
target_train = to_categorical(target_vector_train)
target_test = to_categorical(target_vector_test)

# ニューラルネットワークの作成を開始
network = models.Sequential()

# 活性化関数としてReLUを用いる全結合層を追加
network.add(layers.Dense(units=100,
                         activation="relu",
                         input_shape=(number_of_features,)))

# 活性化関数としてReLUを用いる全結合層を追加
network.add(layers.Dense(units=100, activation="relu"))

# 活性化関数としてソフトマックスを用いる全結合層を追加
network.add(layers.Dense(units=46, activation="softmax"))

# ニューラルネットワークをコンパイル
network.compile(loss="categorical_crossentropy", # クロスエントロピ
                optimizer="rmsprop", # 二乗平均平方根伝搬法
                metrics=["accuracy"]) # 性能指標は精度

# ニューラルネットワークを訓練
history = network.fit(features_train, # 特徴量
                      target_train, # ターゲットベクトル
                      epochs=3, # 3エポック学習
                      verbose=0, # 出力しない
                      batch_size=100, # 1バッチあたりの観測値数
                      validation_data=(features_test, target_test)) # テストデータ

##########

# ターゲット行列を表示
target_train
