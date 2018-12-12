# -*- coding: utf-8 -*-

# ライブラリをロード
import cv2
import numpy as np
from matplotlib import pyplot as plt

# モノクロで画像を読み込み
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# カーネルの作成
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

# 画像をくっきりさせる
image_sharp = cv2.filter2D(image, -1, kernel)

# 画像を表示
plt.imshow(image_sharp, cmap="gray"), plt.axis("off")
plt.show()
