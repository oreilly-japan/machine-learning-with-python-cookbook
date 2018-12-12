# -*- coding: utf-8 -*-

# ライブラリをロード
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

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
target_predicted = classifier.fit(features_train,
    target_train).predict(features_test)

# 混同行列を作成
matrix = confusion_matrix(target_test, target_predicted)

# pandasのDataFrameを作成
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)

# ヒートマップを作成
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues")
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()


