# -*- coding: utf-8 -*-

# ライブラリをロード
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# 特徴量行列とターゲットベクトルを作成
features, target = make_classification(n_samples=10000,
                                       n_features=10,
                                       n_classes=2,
                                       n_informative=3,
                                       random_state=3)

# 訓練セットとテストセットに分割s
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1)

# クラス分類器の作成
logit = LogisticRegression()

# モデルの訓練
logit.fit(features_train, target_train)

# 予測確率の取得
target_probabilities = logit.predict_proba(features_test)[:,1]

# 真陽性率と偽陽性率を計算
false_positive_rate, true_positive_rate, threshold = roc_curve(target_test,
                                                               target_probabilities)

# ROCカーブをプロット
plt.title("Receiver Operating Characteristic")
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

###########

# 予測確率の取得
logit.predict_proba(features_test)[0:1]

###########

logit.classes_

###########

print("閾値:", threshold[116])
print("真陽性率:", true_positive_rate[116])
print("偽陽性率", false_positive_rate[116])

###########

print("閾値:", threshold[45])
print("真陽性率:", true_positive_rate[45])
print("偽陽性率", false_positive_rate[45])

###########

# AUC(カーブの下の面積)を計算
roc_auc_score(target_test, target_probabilities)

