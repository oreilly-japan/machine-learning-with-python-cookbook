# -*- coding: utf-8 -*-

# ライブラリをロード
import cv2
import numpy as np
from matplotlib import pyplot as plt

# モノクロで画像を読み込み
image_bgr = cv2.imread("images/plane_256x256.jpg")
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)

# コーナー検出のパラメータを設定
block_size = 2
aperture = 29
free_parameter = 0.04

# コーナー検出
detector_responses = cv2.cornerHarris(image_gray,
                                      block_size,
                                      aperture,
                                      free_parameter)

# コーナー部分を強調
# Large corner markers
detector_responses = cv2.dilate(detector_responses, None)

# 検出器の反応が閾値以上に大きい場所だけを保持して、白くする
threshold = 0.02
image_bgr[detector_responses >
          threshold *
          detector_responses.max()] = [255,255,255]

# モノクロに変換
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# 画像を表示
plt.imshow(image_gray, cmap="gray"), plt.axis("off")
plt.show()

##########

# コーナーの候補を表示
plt.imshow(detector_responses, cmap='gray'), plt.axis("off")
plt.show()

##########

# 画像をロード
image_bgr = cv2.imread('images/plane_256x256.jpg')
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# 検出したいコーナーの数
corners_to_detect = 10
minimum_quality_score = 0.05
minimum_distance = 25

# コーナーを検出
corners = cv2.goodFeaturesToTrack(image_gray,
                                  corners_to_detect,
                                  minimum_quality_score,
                                  minimum_distance)
corners = np.float32(corners)

# 各コーナーに白い円を描画
for corner in corners:
    x, y = corner[0]
    cv2.circle(image_bgr, (x,y), 10, (255,255,255), -1)

# モノクロで画像を読み込み
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# 画像を表示
plt.imshow(image_rgb, cmap='gray'), plt.axis("off")
plt.show()
