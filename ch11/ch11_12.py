# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# データをロード
iris = datasets.load_iris()

# 特徴量行列を作成
features = iris.data

# ターゲットベクトルを作成
target = iris.target

# ターゲットクラス名のリストを作成
class_names = iris.target_names

# 訓練セットとテストセットに分割
features_train, features_test, target_train, target_test = train_test_split(
    features, target, random_state=1)

# ロジスティック回帰器を作成
classifier = LogisticRegression()

# モデルを訓練して予測
model = classifier.fit(features_train, target_train)
target_predicted = model.predict(features_test)

# クラス分類レポートを作成
print(classification_report(target_test,
                            target_predicted,
                            target_names=class_names))
