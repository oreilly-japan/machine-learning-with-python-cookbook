# -*- coding: utf-8 -*-

# ライブラリをロード
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 画像をロードしてRGBに変換
image_bgr = cv2.imread('images/plane_256x256.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 矩形: 始点x、始点y, 幅、高さ
rectangle = (0, 56, 256, 150)

# マスクの初期値を作成
mask = np.zeros(image_rgb.shape[:2], np.uint8)

# grabCutで用いる一時配列を作成
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# grabCutを実行
cv2.grabCut(image_rgb, # 入力画像
            mask,      # マスク
            rectangle, # 範囲指定の矩形
            bgdModel,  # 背景のための一時配列
            fgdModel,  # 前景のための一時配列
            5,         # 繰り返し回数
            cv2.GC_INIT_WITH_RECT) # 矩形を用いて初期化

# マスクを作成。背景であることが確実もしくは高確率な場所を0に、それ以外を1に
mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

# 画像とマスクを掛け合わせて背景を除去
image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]

# 画像を表示
plt.imshow(image_rgb_nobg), plt.axis("off")
plt.show()

##########

# マスクを表示
plt.imshow(mask, cmap='gray'), plt.axis("off")
plt.show()

##########

# マスクを表示
plt.imshow(mask_2, cmap='gray'), plt.axis("off")
plt.show()
