# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn.naive_bayes import BernoulliNB

# 2値特徴量を作成
features = np.random.randint(2, size=(100, 3))

# 2値ターゲットベクトルを作成
target = np.random.randint(2, size=(100, 1)).ravel()

# 事前確率を指定してベルヌーイナイーブベイズクラス分類器を作成
classifer = BernoulliNB(class_prior=[0.25, 0.5])

# ベルヌーイナイーブベイズクラス分類器を訓練
model = classifer.fit(features, target)

###########

model_uniform_prior = BernoulliNB(class_prior=None, fit_prior=True)
