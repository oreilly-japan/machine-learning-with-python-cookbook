# -*- coding: utf-8 -*-

# ライブラリをロード
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 数字データセットをロード
digits = datasets.load_digits()

# 特徴量行列を作成
features = digits.data

# ターゲットベクトルを作成
target = digits.target

# 標準化器を作成
standardizer = StandardScaler()

# ロジスティック回帰器を作成 
logit = LogisticRegression()

# 標準化を行い、ロジスティック回帰を実行するパイプラインを作成
pipeline = make_pipeline(standardizer, logit)


# k-分割交差検証器を作成
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# k-分割交差検証を実行
cv_results = cross_val_score(pipeline,    # パイプライン
                             features,    # 特徴量行列
                             target,      # ターゲットベクトル
                             cv=kf,       # 交差検証手法
                             scoring="accuracy", # スコア関数  
                             n_jobs=-1)   # すべてのCPUを利用

# 平均を計算
cv_results.mean()

##########

# 10回行った評価のスコアを表示
cv_results

##########

# ライブラリをロード
from sklearn.model_selection import train_test_split

# 訓練セットとテストセットに分割
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1)

# 標準化器を訓練セットだけで訓練
standardizer.fit(features_train)

# 訓練した標準化器を訓練セットとテストセットに適用
features_train_std = standardizer.transform(features_train)
features_test_std = standardizer.transform(features_test)


##########

# パイプラインを作成
pipeline = make_pipeline(standardizer, logit)

##########

# k-分割交差検証を実行
cv_results = cross_val_score(pipeline, # パイプライン
                             features, # 特徴量行列
                             target,   # ターゲットベクトル
                             cv=kf,    # 交差検証手法
                             scoring="accuracy", # スコア関数
                             n_jobs=-1) # すべてのコアを利用
