# -*- coding: utf-8 -*-

# ライブラリをロード
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 画像をグレースケール（モノクロ）として読み込み
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)

##########

# 画像を表示
plt.imshow(image, cmap="gray"), plt.axis("off")
plt.show()

##########

# データ型を表示
type(image)

##########

# 画像データを表示
image

##########

# データの形状を表示
image.shape

##########

# 最初のピクセルを表示
image[0,0]

##########

# カラーで画像を読み込み
image_bgr = cv2.imread("images/plane.jpg", cv2.IMREAD_COLOR)

# ピクセルを表示
image_bgr[0,0]

##########

# RGBに変換
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 画像を表示
plt.imshow(image_rgb), plt.axis("off")
plt.show()
