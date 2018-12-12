# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# ランダムフォレストクラス分類器を作成
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)

# ランダムフォレストクラス分類器を訓練
model = randomforest.fit(features, target)

# 特徴量重要性を計算
importances = model.feature_importances_

# 特徴量重要性を降順にソート
indices = np.argsort(importances)[::-1]

# 特徴量の名前を、ソートした順に並び替え
names = [iris.feature_names[i] for i in indices]

# プロットを作成
plt.figure()

# プロットのタイトルを作成
plt.title("Feature Importance")

# 棒グラフを追加
plt.bar(range(features.shape[1]), importances[indices])

# X軸に特徴量の名前を追加
plt.xticks(range(features.shape[1]), names, rotation=90)

# プロットを表示
plt.show()

##########

# 特徴量重要度を表示
model.feature_importances_
