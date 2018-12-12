# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

# データをロード with only two classes and two features
iris = datasets.load_iris()
features = iris.data[:100,:2]
target = iris.target[:100]

# 特徴量を標準化
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# サポートベクタクラス分類器を作成
svc = LinearSVC(C=1.0)

# サポートベクタクラス分類器を訓練
model = svc.fit(features_standardized, target)

##########

# ライブラリをロード
from matplotlib import pyplot as plt

# 観測値をそれぞれの色でプロット
color = ["black" if c == 0 else "lightgrey" for c in target]
plt.scatter(features_standardized[:,0], features_standardized[:,1], c=color)

# 超平面を作成
w = svc.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2.5, 2.5)
yy = a * xx - (svc.intercept_[0]) / w[1]

# 超平面をプロット
plt.plot(xx, yy)
plt.axis("off"), plt.show();

##########

# 新たな観測値の作成
new_observation = [[ -2,  3]]

# 新たな観測値のクラスを予測
svc.predict(new_observation)


