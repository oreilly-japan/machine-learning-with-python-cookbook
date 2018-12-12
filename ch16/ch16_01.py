# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# データをロード with only two classes
iris = datasets.load_iris()
features = iris.data[:100,:]
target = iris.target[:100]

# 特徴量を標準化
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# ロジスティック回帰器を作成
logistic_regression = LogisticRegression(random_state=0)

# ロジスティック回帰器を訓練
model = logistic_regression.fit(features_standardized, target)

##########

# 新たな観測値を作成
new_observation = [[.5, .5, .5, .5]]

# クラスを予測
model.predict(new_observation)

##########

# 予測確率を表示
model.predict_proba(new_observation)
