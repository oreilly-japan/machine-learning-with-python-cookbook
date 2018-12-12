# -*- coding: utf-8 -*-

# ライブラリをロード
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# 特徴量行列を作成
features = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])

# PolynomialFeatures オブジェクトを作成
polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)

# 多項式特徴量を作成
polynomial_interaction.fit_transform(features)

##########

interaction = PolynomialFeatures(degree=2,
              interaction_only=True, include_bias=False)
interaction.fit_transform(features)


