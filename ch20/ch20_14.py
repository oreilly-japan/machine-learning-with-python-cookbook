# -*- coding: utf-8 -*-

# ライブラリをロード
from keras import models
from keras import layers
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

# ニューラルネットワークの作成を開始
network = models.Sequential()

# 活性化関数としてReLUを用いる全結合層を追加
network.add(layers.Dense(units=16, activation="relu", input_shape=(10,)))

# 活性化関数としてReLUを用いる全結合層を追加
network.add(layers.Dense(units=16, activation="relu"))

# 活性化関数としてシグモイド関数を用いる全結合層を追加
network.add(layers.Dense(units=1, activation="sigmoid"))

# ネットワーク構造を可視化
SVG(model_to_dot(network, show_shapes=True).create(prog="dot", format="svg"))

##########

# 可視化した図をファイルにセーブ
plot_model(network, show_shapes=True, to_file="network.png")

##########

# ネットワーク構造を可視化
SVG(model_to_dot(network, show_shapes=False).create(prog="dot", format="svg"))
