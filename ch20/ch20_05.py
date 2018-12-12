# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# 乱数シードを設定
np.random.seed(0)

# 特徴量行列とターゲットベクトルを作成
features, target = make_regression(n_samples = 10000,
                                   n_features = 3,
                                   n_informative = 3,
                                   n_targets = 1,
                                   noise = 0.0,
                                   random_state = 0)

# 訓練セットとデータセットに分割
features_train, features_test, target_train, target_test = train_test_split(
features, target, test_size=0.33, random_state=0)

# ニューラルネットワークの作成を開始
network = models.Sequential()

# 活性化関数としてReLUを用いる全結合層を追加
network.add(layers.Dense(units=32,
                         activation="relu",
                         input_shape=(features_train.shape[1],)))

# 活性化関数としてReLUを用いる全結合層を追加
network.add(layers.Dense(units=32, activation="relu"))

# 活性化関数を用いない全結合層を追加
network.add(layers.Dense(units=1))

# ニューラルネットワークをコンパイル
network.compile(loss="mse", # 平均2乗誤差を最小化
                optimizer="RMSprop", # 最適化手法を指定
                metrics=["mse"]) # 性能指標は平均2乗誤差

# ニューラルネットワークを訓練
history = network.fit(features_train, # 特徴量
                      target_train, # ターゲットベクトル
                      epochs=10, # エポック数
                      verbose=0, # 出力しない
                      batch_size=100, # 1バッチあたりの観測値数
                      validation_data=(features_test, target_test)) # テストデータ



