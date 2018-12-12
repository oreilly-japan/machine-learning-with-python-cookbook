# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# ロジスティック回帰器を作成
logistic = linear_model.LogisticRegression()

# 正則化ペナルティハイパパラメータの候補となる値の範囲を指定
penalty = ["l1", "l2"]

# 正則化強度ハイパパラメータの候補となる値の範囲を指定
C = np.logspace(0, 4, 1000)

# ハイパパラメータ候補辞書を作成
hyperparameters = dict(C=C, penalty=penalty)

# グリッド探索器を作成
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=1)

# グリッド探索器を訓練
best_model = gridsearch.fit(features, target)

############

# 1コアだけ使用するグリッド探索器を作成 
clf = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=1, verbose=1)

# グリッド探索器を訓練
best_model = clf.fit(features, target)
