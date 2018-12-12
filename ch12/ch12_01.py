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
penalty = ['l1', 'l2']

# 正則化強度ハイパパラメータの候補となる値の範囲を指定
C = np.logspace(0, 4, 10)

# ハイパパラメータ候補辞書を作成
hyperparameters = dict(C=C, penalty=penalty)

# グリッド探索器を作成
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

# グリッド探索器を訓練
best_model = gridsearch.fit(features, target)

###########

np.logspace(0, 4, 10)

###########

# 最良のハイパパラメータを表示
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

###########

# ターゲットベクトルを予測
best_model.predict(features)
