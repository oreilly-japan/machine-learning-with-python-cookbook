# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn import datasets
import numpy as np

# データをロード
digits = datasets.load_digits()

# 特徴量行列の標準化
features = StandardScaler().fit_transform(digits.data)

# 疎行列の作成
features_sparse = csr_matrix(features)

# TSVDの作成
tsvd = TruncatedSVD(n_components=10)

# 疎行列に対してTSVDを実行
features_sparse_tsvd = tsvd.fit(features_sparse).transform(features_sparse)

# 結果を表示
print("もとの特徴量数:", features_sparse.shape[1])
print("削減後の特徴量数:", features_sparse_tsvd.shape[1])

##########

# 最初の3つの成分の寄与率の総計
tsvd.explained_variance_ratio_[0:3].sum()

##########

# 特徴量数-1を指定してTSVDを作成して実行
tsvd = TruncatedSVD(n_components=features_sparse.shape[1]-1)
features_tsvd = tsvd.fit(features)

# 説明された分散の割合のリスト
tsvd_var_ratios = tsvd.explained_variance_ratio_

# 関数を定義
def select_n_components(var_ratio, goal_var):
    # 説明された寄与率の累計変数を初期化
    total_variance = 0.0

    # 特徴量数の初期値
    n_components = 0

    # それぞれの特徴量の寄与率に対して
    for explained_variance in var_ratio:

        # 寄与率を累計に追加
        total_variance += explained_variance

        # 特徴量数に1追加
        n_components += 1

        # 説明された寄与率が目標値に到達しているなら
        if total_variance >= goal_var:
	        # ループを終了
            break

    # 成分(特徴量)数を返す
    return n_components

# 関数を実行
select_n_components(tsvd_var_ratios, 0.95)

