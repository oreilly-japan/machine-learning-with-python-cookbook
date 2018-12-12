# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV, cross_val_score

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# ロジスティック回帰器を作成
logistic = linear_model.LogisticRegression()

# 正則化強度ハイパパラメータの候補値を20個作成
C = np.logspace(0, 4, 20)

# ハイパパラメータ候補辞書を作成
hyperparameters = dict(C=C)

# グリッド探索器を作成
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=0)

# 2重交差検証を行い、平均値を表示
cross_val_score(gridsearch, features, target).mean()

###########

gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=1)

###########

best_model = gridsearch.fit(features, target)

###########

scores = cross_val_score(gridsearch, features, target)



