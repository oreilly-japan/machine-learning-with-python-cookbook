# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# irisデータセットをロード
iris = load_iris()

# 特徴量行列を作成 
features = iris.data

# ターゲットベクトルを作成
target = iris.target

# 最初の40の観測値を削除
features = features[40:,:]
target = target[40:]

# クラス0であるかどうかを示す2値ターゲットベクトルを作成
target = np.where((target == 0), 0, 1)

# バランスの崩れたターゲットベクトルを表示
target

##########

# 重みを作成
weights = {0: .9, 1: 0.1}

# ランダムフォレストクラス分類器を、重みを指定して作成
RandomForestClassifier(class_weight=weights)

##########

# ランダムフォレストクラス分類器を、重みをbalancedに指定して作成
RandomForestClassifier(class_weight="balanced")

##########

# それぞれのクラスの観測値のインデックスを取得
i_class0 = np.where(target == 0)[0]
i_class1 = np.where(target == 1)[0]

# それぞれのクラスの観測値数を計算
n_class0 = len(i_class0)
n_class1 = len(i_class1)

# クラス0のそれぞれの観測値に対して、ランダムに
# クラス1から非復元抽出
i_class1_downsampled = np.random.choice(i_class1, size=n_class0, replace=False)

# クラス0のターゲットベクトルと、
# ダウンサンプリングしたクラス1のターゲットベクトルを結合
np.hstack((target[i_class0], target[i_class1_downsampled]))

##########

# クラス0の特徴量行列と、
# ダウンサンプリングしたクラス1の特徴量行列を結合
np.vstack((features[i_class0,:], features[i_class1_downsampled,:]))[0:5]

##########

# クラス1のそれぞれの観測値に対して、ランダムにクラス0から復元抽出
i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)

# クラス0のアップサンプリングされたターゲットベクトルとクラス1のターゲットベクトルを結合
np.concatenate((target[i_class0_upsampled], target[i_class1]))

##########

# クラス0をアップサンプリングした特徴量行列と、クラス1の特徴量行列を結合
np.vstack((features[i_class0_upsampled,:], features[i_class1,:]))[0:5]



