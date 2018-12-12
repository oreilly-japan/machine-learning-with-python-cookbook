# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression

# ターゲットベクトルと特徴量行列を作成
features, target = make_regression(n_samples = 100,
                                   n_features = 3,
                                   random_state = 1)

# 訓練セットとテストセットに分割
features_train, features_test, target_train, target_test = train_test_split(
     features, target, test_size=0.10, random_state=1)

# 独自の評価基準関数を作成
def custom_metric(target_test, target_predicted):
    # R^2スコアを計算
    r2 = r2_score(target_test, target_predicted)
    # R^2スコアを返す
    return r2

# スコアが高い方がよいと指定して、スコア付けオブジェクトを作成
score = make_scorer(custom_metric, greater_is_better=True)

# リッジ回帰器を作成
classifier = Ridge()

# リッジ回帰モデルを訓練
model = classifier.fit(features_train, target_train)

# 独自のスコア関数を適用
score(model, features_test, target_test)

##########

# 値の予測
target_predicted = model.predict(features_test)

# R^2スコアの計算
r2_score(target_test, target_predicted)
