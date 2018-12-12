# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 乱数シードを設定
np.random.seed(0)

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# StandardScalerとPCAを含む前処理オブジェクトを作成
preprocess = FeatureUnion([("std", StandardScaler()), ("pca", PCA())])

# パイプラインを作成
pipe = Pipeline([("preprocess", preprocess),
                 ("classifier", LogisticRegression())])

# 候補値の空間を作成
search_space = [{"preprocess__pca__n_components": [1, 2, 3],
                 "classifier__penalty": ["l1", "l2"],
                 "classifier__C": np.logspace(0, 4, 10)}]

# グリッド探索器を作成
clf = GridSearchCV(pipe, search_space, cv=5, verbose=0, n_jobs=-1)

# グリッド探索器を訓練
best_model = clf.fit(features, target)

##########

# 最良のモデルを表示
best_model.best_estimator_.get_params()['preprocess__pca__n_components']
