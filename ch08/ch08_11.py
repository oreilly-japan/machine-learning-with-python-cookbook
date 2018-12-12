# -*- coding: utf-8 -*-

# ライブラリをロード
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
image_gray = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# 輝度の中央値を計算
median_intensity = np.median(image_gray)

# 中央値から1標準偏差分上と下を閾値として設定
lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))

# Cannyエッジ検出を適用
image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold)

# 画像を表示
plt.imshow(image_canny, cmap="gray"), plt.axis("off")
plt.show()

##########

