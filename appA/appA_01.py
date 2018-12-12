# -*- coding: utf-8 -*-

# ライブラリをロード
from janome.tokenizer import Tokenizer

# テキストを作成
text = "貴社の記者が汽車で帰社した。"

# トークン化器を作成
t = Tokenizer()

# トークン化
tokens = list(t.tokenize(text))

for t in tokens:
    print(t)

##########

# 表層形を表示
[token.surface for token in tokens]

#########

# 基本形を表示
[token.base_form for token in tokens]





