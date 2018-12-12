# -*- coding: utf-8 -*-

# ライブラリをロード
import cv2
import numpy as np

# モノクロで画像を読み込み
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)

# 画像を保存
cv2.imwrite("images/plane_new.jpg", image)

##########


