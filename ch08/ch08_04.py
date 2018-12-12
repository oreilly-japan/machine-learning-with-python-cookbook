# -*- coding: utf-8 -*-

# ライブラリのロード
import cv2
import numpy as np
from matplotlib import pyplot as plt

# モノクロで画像を読み込み
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# 列の最初の半分を選択。行はすべての行を使う
image_cropped = image[:,:128]

# 画像を表示
plt.imshow(image_cropped, cmap="gray"), plt.axis("off")
plt.show()

