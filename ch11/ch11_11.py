# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve

# データをロード
digits = load_digits()

# 特徴量行列とターゲットベクトルを作成
features, target = digits.data, digits.target

# 訓練セットサイズを変更しながら交差検証を用いた訓練を行い、スコアを取得
train_sizes, train_scores, test_scores = learning_curve(# クラス分類器
                                                        RandomForestClassifier(),
                                                        # 特徴量行列
                                                        features,
                                                        # ターゲットベクトル
                                                        target,
                                                        # 分割数
                                                        cv=10,
                                                        # 性能評価指標
                                                        scoring='accuracy',
                                                        # すべてのコアを利用
                                                        n_jobs=-1,
                                                        # 訓練セットのサイズ
                                                        # を50通りに設定
                                                       train_sizes=np.linspace(
                                                       0.01,
                                                       1.0,
                                                       50))

# 訓練セットスコアの平均と分散を計算
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# テストセットスコアの平均と分散を計算
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# 線を描画
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# 帯領域を描画
plt.fill_between(train_sizes, train_mean - train_std,
                 train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std,
                 test_mean + test_std, color="#DDDDDD")

# プロットを作成
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"),
plt.legend(loc="best")
plt.tight_layout()
plt.show()
