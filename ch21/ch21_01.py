# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.externals import joblib

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# ランダムフォレストクラス分類器を作成
classifer = RandomForestClassifier()

# ランダムフォレストクラス分類器を訓練
model = classifer.fit(features, target)

# 訓練したモデルをピクルファイルとしてセーブ
joblib.dump(model, "model.pkl")

##########

# モデルをファイルからロード
classifer = joblib.load("model.pkl")

##########

# 新たな観測値を作成
new_observation = [[ 5.2,  3.2,  1.1,  0.1]]

# 観測値のクラスを予測
classifer.predict(new_observation)

##########

# ライブラリをロード
import sklearn

# scikit-learnのバージョンを取得
scikit_version = joblib.__version__

# 訓練したモデルをピクルファイルとしてセーブ
joblib.dump(model, "model_{version}.pkl".format(version=scikit_version))
