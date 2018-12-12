# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

# 乱数シードを設定
np.random.seed(0)

# 2つの特徴量を生成
features = np.random.randn(200, 2)

# XORゲート(ここでは分からなくて良い)を用いて、
# 線形分離不能なデータを作成
target_xor = np.logical_xor(features[:, 0] > 0, features[:, 1] > 0)
target = np.where(target_xor, 0, 1)

# 放射基底関数(rbf)カーネルを用いたサポートベクタマシンを作成
svc = SVC(kernel="rbf", random_state=0, gamma=1, C=1)

# クラス分類器を訓練
model = svc.fit(features, target)

##########

# 観測値と決定境界超平面をプロット
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier):
    cmap = ListedColormap(("red", "blue"))
    xx1, xx2 = np.meshgrid(np.arange(-3, 3, 0.02), np.arange(-3, 3, 0.02))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.1, cmap=cmap)

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker="+", label=cl)

##########

# 線形カーネルを用いたサポートベクタクラス分類器を作成
svc_linear = SVC(kernel="linear", random_state=0, C=1)

# クラス分類器を訓練
svc_linear.fit(features, target)

##########

# 観測値と超平面をプロット
plot_decision_regions(features, target, classifier=svc_linear)
plt.axis("off"), plt.show();

##########

# 放射基底関数カーネルを用いたサポートベクタクラス分類器を作成
svc = SVC(kernel="rbf", random_state=0, gamma=1, C=1)

# クラス分類器を訓練
model = svc.fit(features, target)

###########

# 観測値と超平面をプロット
plot_decision_regions(features, target, classifier=svc)
plt.axis("off"), plt.show();
