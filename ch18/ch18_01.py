# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# ガウシアンナイーブベイズクラス分類器を作成
classifer = GaussianNB()

# ガウシアンナイーブベイズクラス分類器を訓練
model = classifer.fit(features, target)

##########

# 新たな観測値を作成
new_observation = [[ 4,  4,  4,  0.4]]

# クラスを予測
model.predict(new_observation)

##########

# ガウシアンナイーブベイズクラス分類器を、個々のクラスの
# 確率を指定して作成
clf = GaussianNB(priors=[0.25, 0.25, 0.5])

# ガウシアンナイーブベイズクラス分類器を訓練
model = classifer.fit(features, target)
