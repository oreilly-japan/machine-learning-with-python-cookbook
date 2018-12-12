# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 人工的な特徴量行列を作成
features, _ = make_blobs(n_samples = 50,
                         n_features = 2,
                         centers = 3,
                         random_state = 1)

# DataFrameを作成
dataframe = pd.DataFrame(features, columns=["feature_1", "feature_2"])

# k-meansクラスタ分け器を作成
clusterer = KMeans(3, random_state=0)

# クラスタ分け器を訓練
clusterer.fit(features)

# クラスタ分けを実行
dataframe["group"] = clusterer.predict(features)

# 観測値の最初の数個を表示
dataframe.head(5)
