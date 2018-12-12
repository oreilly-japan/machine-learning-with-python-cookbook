# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

# データをロード
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 標準化器を作成
standardizer = StandardScaler()

# 特徴量を標準化
X_std = standardizer.fit_transform(X)

# 近傍値数を5に指定してKNNクラス分類器を訓練
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_std, y)

# 2つの観測値を作成
new_observations = [[ 0.75,  0.75,  0.75,  0.75],
                    [ 1,  1,  1,  1]]

# 2つの観測値のクラスを予測
knn.predict(new_observations)

##########

# それぞれの観測値が3つのクラスに属する確率を表示
knn.predict_proba(new_observations)

##########

knn.predict(new_observations)

