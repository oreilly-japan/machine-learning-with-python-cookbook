# -*- coding: utf-8 -*-

# ライブラリをロード
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image
from sklearn import tree

# データをロード
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 決定木回帰器オブジェクトを作成
decisiontree = DecisionTreeClassifier(random_state=0)

# 回帰器を訓練
model = decisiontree.fit(features, target)

# DOTフォーマットでデータを作成
dot_data = tree.export_graphviz(decisiontree,
                                out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names)

# グラフを描画
graph = pydotplus.graph_from_dot_data(dot_data)

# グラフを表示
Image(graph.create_png())

##########

# PDFを作成
graph.write_pdf("iris.pdf")

##########

# PNGを作成
graph.write_png("iris.png")
