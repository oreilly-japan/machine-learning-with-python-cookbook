# -*- coding: utf-8 -*-

# ライブラリをロード
from scipy.stats import uniform
from sklearn import linear_model, datasets
from sklearn.model_selection import RandomizedSearchCV

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# ロジスティック回帰器を作成
logistic = linear_model.LogisticRegression()

# 正則化ペナルティハイパパラメータの候補となる値の範囲を指定
penalty = ['l1', 'l2']

# 正則化強度ハイパパラメータの候補となる値の範囲を指定
C = uniform(loc=0, scale=4)

# ハイパパラメータ候補辞書を作成
hyperparameters = dict(C=C, penalty=penalty)

# ランダム探索器を作成
randomizedsearch = RandomizedSearchCV(
    logistic, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0,
    n_jobs=-1)

# ランダム探索器を訓練
best_model = randomizedsearch.fit(features, target)

###########

# 0から4までの一様分布を作り、10点をサンプリングする
uniform(loc=0, scale=4).rvs(10)

###########

# 最良のハイパパラメータを表示
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

###########

# ターゲットベクトルを予測
best_model.predict(features)
