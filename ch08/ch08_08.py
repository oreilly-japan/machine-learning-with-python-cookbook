# -*- coding: utf-8 -*-

# ライブラリをロード
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 画像をロード
image_bgr = cv2.imread('images/plane_256x256.jpg')

# BGR色空間からHSV色空間に変換
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# HSV空間で「青」の範囲を定義
lower_blue = np.array([50,100,50])
upper_blue = np.array([130,255,255])

# マスクを作成
mask = cv2.inRange(image_hsv, lower_blue, upper_blue)

# 画像にマスクを適用
image_bgr_masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

# BGR色空間からRGB色空間に変換
image_rgb = cv2.cvtColor(image_bgr_masked, cv2.COLOR_BGR2RGB)

# 画像を表示
plt.imshow(image_rgb), plt.axis("off")
plt.show()

##########

# マスク画像を表示
# Show image
plt.imshow(mask, cmap='gray'), plt.axis("off")
plt.show()



