# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

# 簡単な人工データを生成
features, _ = make_blobs(n_samples = 10,
                         n_features = 2,
                         centers = 1,
                         random_state = 1)

# 最初の特徴量の値を極端な値に置換
features[0,0] = 10000
features[0,1] = 10000

# 検出器を作成
outlier_detector = EllipticEnvelope(contamination=.1)

# 検出器を訓練
outlier_detector.fit(features)

# 外れ値を予測
outlier_detector.predict(features)

##########

# 特徴量を1つ作成
feature = features[:,0]

# 外れ値のインデックスを返す関数を作る
def indicies_of_outliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((x > upper_bound) | (x < lower_bound))

# 関数を実行
indicies_of_outliers(feature)
