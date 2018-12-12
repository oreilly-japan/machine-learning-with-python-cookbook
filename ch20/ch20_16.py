# -*- coding: utf-8 -*-

#  このサンプルコードは、文中のままでは動作しないので、動作に必要な
#  コードを訳者が補った。
#  https://github.com/chrisalbon/simulated_datasets
#  のimagesディレクトリをダウンロードすれば動作する。
#  このディレクトリにはmnistから抽出した0と1がそれぞれ別の
#  ディレクトリに分けて置かれている。
#  ここに示したニューラルネットは20.15のニューラルネットを
#  改変し2クラス分類問題用としている。
#
#  注意： processed/images ディレクトリが無いと動作しないので
#         実行前にこのディレクトリを作成すること。

#
from keras.datasets import mnist
# 画像情報を設定
channels = 1
height = 28
width = 28

(data_train, target_train), (data_test, target_test) = mnist.load_data()

# 訓練画像の形状を変更して特徴量に変換
data_train = data_train.reshape(data_train.shape[0], channels, height, width)

####################################

# ライブラリをロード
from keras.preprocessing.image import ImageDataGenerator

# 画像拡張器を作成
augmentation = ImageDataGenerator(featurewise_center=True, # ZCA ホワイトニング
                                  zoom_range=0.3, # ランダムに画像をズーム
                                  width_shift_range=0.2, # ランダムに画像をシフト
                                  horizontal_flip=True, # ランダムに画像の上下左右入れ替え
                                  data_format='channels_first', # ネットワークに合わせて追加
                                  rotation_range=90) # ランダムに画像を回転

# 画像拡張器が標準化に用いるデータを与えるために、mnistのデータを提示（追加）
augmentation.fit(data_train)

# 'raw/images'ディレクトリにある全ての画像を処理
augment_images = augmentation.flow_from_directory("raw/images", # 画像ディレクトリ
                                                  batch_size=32, # バッチサイズ
                                                  class_mode="binary", # クラス
                                                  target_size=(28, 28), # サイズ(追加)
                                                  color_mode='grayscale') # モノクロ(追加)
                                                  save_to_dir="processed/images")

##########

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

# 活性化関数としてシグモイド関数を用いる全結合層を追加
network.add(Dense(1, activation="sigmoid"))

# ニューラルネットワークをコンパイル
network.compile(loss="binary_crossentropy", # クロスエントロピ
                optimizer="rmsprop", # 二乗平均平方根伝搬法
                metrics=["accuracy"]) # 性能指標は精度

##########
# flow_from_directory はディレクトリ内のデータを使い尽くすとStopIterationを
# を返して停止する。また、fit_generator は、flow_from_directoryが返すジェネレータ
# を直接与えると、stepの指定を無視してディレクトリ内のデータ数で決まるステップ数で
# 停止するので、無限にデータを生成し続けるようにラップする。

def infinite(gen):
    while True:
        for i in gen:
            yield i

import itertools
augment_images = itertools.cycle(augment_images)

# テスト用にデータ生成器を作成。画像拡張は行わない
noaugmentation = ImageDataGenerator(featurewise_center=True, # ZCA ホワイトニング
                                  data_format='channels_first') # ネットワークに合わせて追加
noaugmentation.fit(data_train)

# 'raw/images'ディレクトリにある全ての画像を処理
augment_images_test = noaugmentation.flow_from_directory("raw/images", # 画像ディレクトリ
                                                  batch_size=32, # バッチサイズ
                                                  class_mode="binary", # クラス
                                                  target_size=(28, 28), # サイズ(追加)
                                                  color_mode='grayscale') # モノクロ(追加)

augment_images_test = infinite(augment_images_test)
##########

# ニューラルネットワークを訓練
network.fit_generator(augment_images,
                      # データ生成器を呼び出すエポックあたりの回数
                      steps_per_epoch=2000,
                      # エポック数
                      epochs=5,
                      # テストデータ生成器
                      validation_data=augment_images_test,
                      # エポックごとのテストで生成器を呼び出す回数
                      validation_steps=800) 
