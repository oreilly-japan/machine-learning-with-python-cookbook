# -*- coding: utf-8 -*-

# テキストを生成
text_data = ["   Interrobang. By Aishwarya Henriette     ",
             "Parking And Going. By Karl Gautier",
             "    Today Is The night. By Jarek Prakash   "]

# ホワイトスペースを削除
strip_whitespace = [string.strip() for string in text_data]

# テキストを表示
strip_whitespace

##########

# ピリオドを削除
remove_periods = [string.replace(".", "") for string in strip_whitespace]

# テキストを表示
remove_periods

##########

# 関数を定義
def capitalizer(string: str) -> str:
    return string.upper()

# 関数を適用
[capitalizer(string) for string in remove_periods]

##########

# ライブラリをロード
import re

# 関数を定義
def replace_letters_with_X(string: str) -> str:
    return re.sub(r"[a-zA-Z]", "X", string)

# 関数を適用
[replace_letters_with_X(string) for string in remove_periods]
