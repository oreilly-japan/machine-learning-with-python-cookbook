# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# データフォーマットをカラーチャネルが最初にくるものに指定
K.set_image_data_format("channels_first")

# 乱数シードを設定
np.random.seed(0)

# 画像情報を設定
channels = 1
height = 28
width = 28

# MNISTデータセットのデータとターゲットをロード
(data_train, target_train), (data_test, target_test) = mnist.load_data()

# 訓練画像の形状を変更して特徴量に変換
data_train = data_train.reshape(data_train.shape[0], channels, height, width)

# テスト画像の形状を変更して特徴量に変換
data_test = data_test.reshape(data_test.shape[0], channels, height, width)

# ピクセルの輝度を0から1の間に変換
features_train = data_train / 255
features_test = data_test / 255

# ターゲットをワンホットエンコード
target_train = np_utils.to_categorical(target_train)
target_test = np_utils.to_categorical(target_test)
number_of_classes = target_test.shape[1]

# ニューラルネットワークの作成を開始
network = Sequential()

# 64フィルタ、5x5の窓を持ち、活性化関数としてReLUを用いるコンボリューション層を追加
network.add(Conv2D(filters=64,
                   kernel_size=(5, 5),
                   input_shape=(channels, width, height),
                   activation='relu'))

# 2x2の窓を持つプーリング層を追加
network.add(MaxPooling2D(pool_size=(2, 2)))

# ドロップアウト層を追加
network.add(Dropout(0.5))

# 入力をベクトルにする層を追加
network.add(Flatten())

# 活性化関数としてReLUを用いる128ユニットの全結合層を追加
network.add(Dense(128, activation="relu"))

# ドロップアウト層を追加
network.add(Dropout(0.5))

# 活性化関数としてソフトマックス関数を用いる全結合層を追加
network.add(Dense(number_of_classes, activation="softmax"))

# ニューラルネットワークをコンパイル
network.compile(loss="categorical_crossentropy", # クロスエントロピ
                optimizer="rmsprop", # 二乗平均平方根伝搬法
                metrics=["accuracy"]) # 性能指標は精度

# ニューラルネットワークを訓練
network.fit(features_train, # 特徴量
            target_train, # Target
            epochs=2, # エポック数
            verbose=0, # エポックごとの出力を抑制
            batch_size=1000, # 1バッチあたりの観測値数
            validation_data=(features_test, target_test)) # 評価用データ
