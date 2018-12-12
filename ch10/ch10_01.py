# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold

# テスト用のデータをロード
iris = datasets.load_iris()

# 特徴量とターゲットを作成
features = iris.data
target = iris.target

# 閾値を作成
thresholder = VarianceThreshold(threshold=.5)

# 分散の大きい特徴量行列を作成
features_high_variance = thresholder.fit_transform(features)

# 分散の大きい特徴量行列を表示
features_high_variance[0:3]

##########

# 分散を表示
thresholder.fit(features).variances_

##########

# ライブラリをロード
from sklearn.preprocessing import StandardScaler

# 特徴量行列を標準化
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# それぞれの特徴量を算出
selector = VarianceThreshold()
selector.fit(features_std).variances_
