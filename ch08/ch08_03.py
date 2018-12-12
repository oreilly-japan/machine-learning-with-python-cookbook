# -*- coding: utf-8 -*-

# ライブラリのロード
import cv2
import numpy as np
from matplotlib import pyplot as plt

# モノクロで画像を読み込み
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# 画像を50x50ピクセルにサイズ変更
image_50x50 = cv2.resize(image, (50, 50))

# 画像を表示
plt.imshow(image_50x50, cmap="gray"), plt.axis("off")
plt.show()
