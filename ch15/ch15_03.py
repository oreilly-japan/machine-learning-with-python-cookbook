# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 標準化器を作成
standardizer = StandardScaler()

# 特徴量を標準化
features_standardized = standardizer.fit_transform(features)

# 最近傍法クラス分類器を作成
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

# パイプラインを作成
pipe = Pipeline([("standardizer", standardizer), ("knn", knn)])

# 候補値の空間を作成
search_space = [{"knn__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]

# グリッドサーチを作成
classifier = GridSearchCV(
    pipe, search_space, cv=5, verbose=0).fit(features_standardized, target)


##########

# 最良の近傍点数 (k)
classifier.best_estimator_.get_params()["knn__n_neighbors"]
