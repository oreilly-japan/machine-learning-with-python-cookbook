# -*- coding: utf-8 -*-

# ライブラリをロード
import cv2
import numpy as np
from matplotlib import pyplot as plt

# モノクロで画像を読み込み
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# 画像をぼかす
image_blurry = cv2.blur(image, (5,5))

# 画像を表示
plt.imshow(image_blurry, cmap="gray"), plt.axis("off")
plt.show()

##########

# 画像をぼかす
image_very_blurry = cv2.blur(image, (100,100))

# 画像を表示
plt.imshow(image_very_blurry, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()

##########

# カーネルを作成
kernel = np.ones((5,5)) / 25.0

# カーネルを表示
kernel

##########

# カーネルを適用
image_kernel = cv2.filter2D(image, -1, kernel)

# 画像を表示
plt.imshow(image_kernel, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()
