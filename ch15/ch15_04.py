# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 標準化器を作成
standardizer = StandardScaler()

# 特徴量を標準化
features_standardized = standardizer.fit_transform(features)

# 半径を用いる近傍法クラス分類器を訓練
rnn = RadiusNeighborsClassifier(
    radius=.5, n_jobs=-1).fit(features_standardized, target)

# 観測値を作成
new_observations = [[ 1,  1,  1,  1]]

# 観測値のクラスを予測
rnn.predict(new_observations)
