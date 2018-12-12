# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# ガウシアンナイーブベイズクラス分類器を作成
classifer = GaussianNB()

# シグモイド較正を指定して、交差検証較正クラス分類器を作成
classifer_sigmoid = CalibratedClassifierCV(classifer, cv=2, method='sigmoid')

# 予測確率を較正
classifer_sigmoid.fit(features, target)

# 新たな観測値を作成
new_observation = [[ 2.6,  2.6,  2.6,  0.4]]

# 較正済みの予測確率を表示
classifer_sigmoid.predict_proba(new_observation)

##########

# ガウシアンナイーブベイズを訓練して予測確率を算出
classifer.fit(features, target).predict_proba(new_observation)

##########

# 較正後の予測確率を表示
classifer_sigmoid.predict_proba(new_observation)
