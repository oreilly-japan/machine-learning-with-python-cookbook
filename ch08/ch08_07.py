# -*- coding: utf-8 -*-

# ライブラリをロード
import cv2
import numpy as np
from matplotlib import pyplot as plt

# モノクロで画像を読み込み
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# 画像を強調
image_enhanced = cv2.equalizeHist(image)

# 画像を表示
plt.imshow(image_enhanced, cmap="gray"), plt.axis("off")
plt.show()

##########

# 画像をロード
image_bgr = cv2.imread("images/plane.jpg")

# YUVに変換
image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)

# ヒストグラム均等化を適用
image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])

# RGBに変換
image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

# 画像を表示
plt.imshow(image_rgb), plt.axis("off")
plt.show()
