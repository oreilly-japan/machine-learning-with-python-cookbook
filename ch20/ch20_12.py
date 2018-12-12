# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# 乱数シードを設定
np.random.seed(0)

# 特徴量の数を設定
number_of_features = 100

# 特徴量行列とターゲットベクトルを作成
features, target = make_classification(n_samples = 10000,
                                       n_features = number_of_features,
                                       n_informative = 3,
                                       n_redundant = 0,
                                       n_classes = 2,
                                       weights = [.5, .5],
                                       random_state = 0)

# コンパイル済みのネットワークを返す関数を作成
def create_network():

    # ニューラルネットワークの作成を開始
    network = models.Sequential()

    # 活性化関数としてReLUを用いる全結合層を追加
    network.add(layers.Dense(units=16, activation="relu", input_shape=(
        number_of_features,)))

    # 活性化関数としてReLUを用いる全結合層を追加
    network.add(layers.Dense(units=16, activation="relu"))

    # 活性化関数としてシグモイド関数を用いる全結合層を追加
    network.add(layers.Dense(units=1, activation="sigmoid"))

    # ニューラルネットワークをコンパイル
    network.compile(loss="binary_crossentropy", # クロスエントロピ
                    optimizer="rmsprop", # 二乗平均平方根伝搬法
                    metrics=["accuracy"]) # 性能指標は精度

    # コンパイルしたネットワークをリターン
    return network

# scikit-learnから利用できるようKerasモデルをラップ
neural_network = KerasClassifier(build_fn=create_network,
                                 epochs=10,
                                 batch_size=100,
                                 verbose=0)

# 3分割交差検証を用いてニューラルネットワークを評価
cross_val_score(neural_network, features, target, cv=3)
