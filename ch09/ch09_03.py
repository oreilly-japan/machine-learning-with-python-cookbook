# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Irisデータセットをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# LDAを作成して実行し、更にそれを用いて特徴量を変換
lda = LinearDiscriminantAnalysis(n_components=1)
features_lda = lda.fit(features, target).transform(features)

# 特徴量数を表示
print("もとの特徴量数:", features.shape[1])
print("削減後の特徴量数:", features_lda.shape[1])

##########

lda.explained_variance_ratio_

##########

# LDAを作成して実行
lda = LinearDiscriminantAnalysis(n_components=None)
features_lda = lda.fit(features, target)

# 寄与率の配列を作成
lda_var_ratios = lda.explained_variance_ratio_

# 関数を定義
def select_n_components(var_ratio, goal_var: float) -> int:
    # 累計寄与率を初期化
    total_variance = 0.0

    # 特徴量数の初期値を設定
    n_components = 0

    # それぞれの特徴量の寄与率に対して
    for explained_variance in var_ratio:

        # 寄与率を累計寄与率に加算
        total_variance += explained_variance

        # 主成分数に1足す
        n_components += 1

        # 累計寄与率が目標値に達していたら
        if total_variance >= goal_var:
            # ループ終了
            break

    # 主成分数を返す
    return n_components

# 関数を実行
select_n_components(lda_var_ratios, 0.95)
