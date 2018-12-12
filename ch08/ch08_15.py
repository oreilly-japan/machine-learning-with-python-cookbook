# -*- coding: utf-8 -*-

# ライブラリをロード
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 画像をロード
image_bgr = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_COLOR)

# RGBに変換
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 特徴量値を格納するリストを作成
features = []

# それぞれの色チャンネルに対してヒストグラムを計算
colors = ("r","g","b")

# それぞれの色チャンネルに対してヒストグラムを計算して特徴量値のリストに追加
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb],  # 画像
                        [i],               # チャンネルのインデックス
                        None,              # マスクは用いない
                        [256],             # ヒストグラムの大きさ
                        [0,256])           # 範囲
    features.extend(histogram)

# 観測値の特徴量値となるベクタを作成
observation = np.array(features).flatten()

# 観測値の最初の5つの特徴量を表示
# Show the observation's value for the first five features
observation[0:5]

##########

# RGBチャンネル値を表示
image_rgb[0,0]

##########

# pandasをインポート
import pandas as pd

# データを作成
data = pd.Series([1, 1, 2, 2, 3, 3, 3, 4, 5])

# ヒストグラムを表示
data.hist(grid=False)
plt.show()

##########

# それぞれの色チャンネルに対してヒストグラムを計算
colors = ("r","g","b")

# それぞれの色チャンネルに対して： ヒストグラムを計算してプロット
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb],  # 画像
                        [i],               # チャンネルのインデックス
                        None,              # マスクは用いない
                        [256],             # ヒストグラムの大きさ
                        [0,256])           # 範囲
    plt.plot(histogram, color = channel)
    plt.xlim([0,256])

# プロットしたものを表示
plt.show()

