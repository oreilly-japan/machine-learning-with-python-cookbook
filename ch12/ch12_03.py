# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# 乱数シードを設定
np.random.seed(0)

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# パイプラインを作成
pipe = Pipeline([("classifier", RandomForestClassifier())])

# 候補学習アルゴリズムとそのハイパパラメータの辞書を作成
search_space = [{"classifier": [LogisticRegression()],
                 "classifier__penalty": ['l1', 'l2'],
                 "classifier__C": np.logspace(0, 4, 10)},
                {"classifier": [RandomForestClassifier()],
                 "classifier__n_estimators": [10, 100, 1000],
                 "classifier__max_features": [1, 2, 3]}]

# グリッド探索器を作成
gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0)

# グリッド探索器を訓練
best_model = gridsearch.fit(features, target)

##########

# 最良のモデルを表示
best_model.best_estimator_.get_params()["classifier"]

##########

# ターゲットベクトルを予測
best_model.predict(features)
