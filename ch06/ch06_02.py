# -*- coding: utf-8 -*-

# ライブラリをロード
from bs4 import BeautifulSoup

# HTMLテキストを作成
html = """
       <div class='full_name'><span style='font-weight:bold'>
       Masego</span> Azra</div>
       """

# HTMLをパース
soup = BeautifulSoup(html, "lxml")

# classが"full_name"となっているdivを見つけて、そのテキストを表示
soup.find("div", { "class" : "full_name" }).text.strip()


