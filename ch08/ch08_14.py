# -*- coding: utf-8 -*-

# ライブラリをロード
import cv2
import numpy as np
from matplotlib import pyplot as plt

# BGRで画像をロード
image_bgr = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_COLOR)

# 各チャンネルの平均を計算
channels = cv2.mean(image_bgr)

# 青と赤の値を入れ替え(BGRからRGBへ変換)
observation = np.array([[(channels[2], channels[1], channels[0])]])

# 平均色を表示
observation

##########

# 画像を表示
plt.imshow(observation / 255.0), plt.axis("off")
plt.show()
