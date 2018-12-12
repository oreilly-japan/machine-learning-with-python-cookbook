# -*- coding: utf-8 -*-

# ライブラリをロード
import unicodedata
import sys

# テキストを作成
text_data = ['Hi!!!! I. Love. This. Song....',
             '10000% Agree!!!! #LoveIT',
             'Right?!?!']

# 句読点文字を含む辞書を作成
punctuation = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P'))

# それぞれの文字列から、句読点文字をすべて除去
[string.translate(punctuation) for string in text_data]
