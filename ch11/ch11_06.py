# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# ターゲットベクトルと特徴量行列を作成
features, target = make_classification(n_samples = 10000,
                           n_features = 3,
                           n_informative = 3,
                           n_redundant = 0,
                           n_classes = 3,
                           random_state = 1)

# ロジスティック回帰器を作成
logit = LogisticRegression()

# 精度を用いて交差検証
cross_val_score(logit, features, target, scoring='accuracy')

##########

# マクロ平均F1スコアを用いて交差検証
cross_val_score(logit, features, target, scoring='f1_macro')


