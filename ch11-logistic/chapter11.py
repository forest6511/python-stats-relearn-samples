"""第11章 ロジスティック回帰
『Pythonで学び直す統計・確率』サンプルコード。
Google Colab にこのファイルの内容を貼り付けて実行できます。
収録 LESSON: LESSON 54, LESSON 55, LESSON 56
"""

# --- 共通の準備(序章のフォント設定セル相当) ---
# Colab では 1 回だけ実行すれば OK。日本語ラベルの豆腐(□)化を防ぐ。
!pip install -q matplotlib-fontja
import matplotlib_fontja  # noqa: F401
import matplotlib.pyplot as plt


# ===== LESSON 54 直線ではみ出す　S字カーブの出番 =====

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

rng = np.random.default_rng(42)
n = 500
riyou = rng.integers(0, 30, n)                 # 月の利用回数
# 利用が多いほど退会しにくい設定で、退会(1)/継続(0)を作る
kakuritsu = 1 / (1 + np.exp(-(-1.0 - 0.15 * riyou)))
taikai = rng.binomial(1, kakuritsu)            # 0か1の結果
df = pd.DataFrame({"退会": taikai, "利用回数": riyou})

model = smf.logit("退会 ~ 利用回数", data=df).fit(disp=0)
print(f"切片: {model.params['Intercept']:.3f}")
print(f"利用回数の係数: {model.params['利用回数']:.3f}")
print(f"退会率(全体): {df['退会'].mean():.3f}")

tsukau = pd.DataFrame({"利用回数": [5, 25]})
yosoku = model.predict(tsukau)
for kaisuu, kakuritsu in zip([5, 25], yosoku):
    print(f"利用回数{kaisuu}回の人の退会確率: {kakuritsu:.3f}")

# ===== LESSON 55 係数を読み解く　オッズ比 =====

import numpy as np

odds_hi = np.exp(model.params["利用回数"])
print(f"利用回数の係数: {model.params['利用回数']:.3f}")
print(f"利用回数のオッズ比(exp した値): {odds_hi:.3f}")

for p in [0.1, 0.5, 0.9]:
    odds = p / (1 - p)                 # 確率をオッズに直す
    print(f"確率{p}  →  オッズ {odds:.2f}(起こる:起こらない = {odds:.2f}:1)")

# ===== LESSON 56 的中率90%の罠　混同行列としきい値 =====

pred_kakuritsu = model.predict(df)           # 各人の退会確率
yosoku = (pred_kakuritsu >= 0.5).astype(int)  # 0.5以上なら退会(1)と予測
tekichuu = (yosoku == df["退会"]).mean()

zenin_keizoku = (df["退会"] == 0).mean()      # 全員「退会しない(0)」の的中率

print(f"モデルの的中率(しきい値0.5): {tekichuu:.3f}")
print(f"全員「退会しない」と予測した的中率: {zenin_keizoku:.3f}")

taikai_sha = df["退会"] == 1
tukamaeta = ((yosoku == 1) & taikai_sha).sum()   # 退会を当てた人数
print(f"実際の退会者: {taikai_sha.sum()}人")
print(f"そのうちモデルが捕まえた退会者: {tukamaeta}人")

for shikii in [0.5, 0.2, 0.1]:
    yosoku = (pred_kakuritsu >= shikii).astype(int)
    tukamaeta = ((yosoku == 1) & taikai_sha).sum()
    minogashi = ((yosoku == 0) & taikai_sha).sum()
    print(f"しきい値{shikii}: 退会者44人中 捕捉{tukamaeta}人 "
          f"見逃し{minogashi}人")

