# -*- coding: utf-8 -*-

# ライブラリをロード
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve

# データをロード
digits = load_digits()

# 特徴量行列とターゲットベクトルを生成
features, target = digits.data, digits.target

# パラメータ値の範囲を指定
param_range = np.arange(1, 250, 2)

# 指定したパラメータの範囲の範囲に対して、訓練精度とテスト精度を計算
train_scores, test_scores = validation_curve(
    # クラス分類器
    RandomForestClassifier(),
    # 特徴量行列
    features,
    # ターゲットベクトル
    target,
    # 変更するハイパパラメータ
    param_name="n_estimators",
    # ハイパパラメータ値の範囲
    param_range=param_range,
    # 交差検証の分割数
    cv=3,
    # 性能評価基準
    scoring="accuracy",
    # すべてのコアを利用
    n_jobs=-1)

# 訓練セットスコアの平均と標準偏差を算出
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# テストセットスコアの平均と標準偏差を算出
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# 訓練セットとテストセットの平均精度スコアをプロット
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

# 訓練セットとテストセットの精度帯をプロット
plt.fill_between(param_range, train_mean - train_std,
                 train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std,
                 test_mean + test_std, color="gainsboro")

# プロットを作成
plt.title("Validation Curve With Random Forest")
plt.xlabel("Number Of Trees")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()
