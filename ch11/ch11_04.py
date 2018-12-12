# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 特徴量行列とターゲットベクトルを作成
X, y = make_classification(n_samples = 10000,
                           n_features = 3,
                           n_informative = 3,
                           n_redundant = 0,
                           n_classes = 2,
                           random_state = 1)

# ロジスティック回帰器を作成
logit = LogisticRegression()

# 精度をスコアとして交差検証
cross_val_score(logit, X, y, scoring="accuracy")

##########

# 適合率(precision)を用いて交差検証
cross_val_score(logit, X, y, scoring="precision")

##########

# 再現率(recall)を用いて交差検証
cross_val_score(logit, X, y, scoring="recall")

##########

# f1を用いて交差検証
cross_val_score(logit, X, y, scoring="f1")

##########

# ライブラリをロード
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    random_state=1)

# テストターゲットベクトルに対して予測
y_hat = logit.fit(X_train, y_train).predict(X_test)

# 精度を計算
accuracy_score(y_test, y_hat)
