# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

# データをロード
iris = load_iris()
features = iris.data
target = iris.target

# データを整数に変換して、カテゴリデータとして扱う
features = features.astype(int)

# カイ二乗統計量が最大の2つの特徴量を選択
chi2_selector = SelectKBest(chi2, k=2)
features_kbest = chi2_selector.fit_transform(features, target)

# 結果を表示
print("もとの特徴量数:", features.shape[1])
print("削減後の特徴量数:", features_kbest.shape[1])

##########

# もっとも高いF-値を持つ特徴量を2つ選択
fvalue_selector = SelectKBest(f_classif, k=2)
features_kbest = fvalue_selector.fit_transform(features, target)

# 結果を表示
print("もとの特徴量数:", features.shape[1])
print("削減後の特徴量数:", features_kbest.shape[1])

##########

# ライブラリをロード
from sklearn.feature_selection import SelectPercentile

# F値が上位75パーセントの特徴量を選択
fvalue_selector = SelectPercentile(f_classif, percentile=75)
features_kbest = fvalue_selector.fit_transform(features, target)

# 結果を表示
print("もとの特徴量数:", features.shape[1])
print("削減後の特徴量数:", features_kbest.shape[1])

