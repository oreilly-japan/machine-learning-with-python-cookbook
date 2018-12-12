# -*- coding: utf-8 -*-

# ライブラリをロード
import cv2
import numpy as np
from matplotlib import pyplot as plt

# モノクロで画像読み込み
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# 10x10ピクセルにサイズ変換
image_10x10 = cv2.resize(image, (10, 10))

# 1次元ベクトルに変換
image_10x10.flatten()

##########

plt.imshow(image_10x10, cmap="gray"), plt.axis("off")
plt.show()

##########

image_10x10.shape

##########

image_10x10.flatten().shape

##########

# カラーで画像をロード
image_color = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_COLOR)

# 10x10ピクセルにサイズ変換
image_color_10x10 = cv2.resize(image_color, (10, 10))

# 1次元のベクトルに変換して、形を表示
image_color_10x10.flatten().shape

##########

# モノクロで画像を読み込み
image_256x256_gray = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# 1次元のベクトルに変換して、形を表示
image_256x256_gray.flatten().shape
