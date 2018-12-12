# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.feature_selection import SelectFromModel

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# ランダムフォレストクラス分類器を作成
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)

# 重要度が閾値以上の特徴量だけを選択するオブジェクトを生成
selector = SelectFromModel(randomforest, threshold=0.3)

# 選択器を用いて新たな特徴量行列を作成
features_important = selector.fit_transform(features, target)

# 選択された重要度の高い特徴量を用いてランダムフォレストを訓練
model = randomforest.fit(features_important, target)
