"""第10章 重回帰分析
『Pythonで学び直す統計・確率』サンプルコード。
Google Colab にこのファイルの内容を貼り付けて実行できます。
収録 LESSON: LESSON 50, LESSON 51, LESSON 52, LESSON 53
"""

# --- 共通の準備(序章のフォント設定セル相当) ---
# Colab では 1 回だけ実行すれば OK。日本語ラベルの豆腐(□)化を防ぐ。
!pip install -q matplotlib-fontja
import matplotlib_fontja  # noqa: F401
import matplotlib.pyplot as plt


# ===== LESSON 50 要因は1つじゃない　重回帰への拡張 =====

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

rng = np.random.default_rng(42)
n = 100
hirosa = rng.uniform(20, 70, n)          # 広さ(平米)
chikunen = rng.uniform(0, 40, n)         # 築年数
eki = rng.uniform(1, 15, n)              # 駅からの徒歩(分)
# 家賃を「広さ・築年数・駅距離＋少しの誤差」から作る
yachin = 1.5 * hirosa - 0.3 * chikunen - 0.8 * eki + 40 + rng.normal(0, 3, n)
df = pd.DataFrame({"家賃": yachin, "広さ": hirosa,
                   "築年数": chikunen, "駅距離": eki})

tan = smf.ols("家賃 ~ 広さ", data=df).fit()               # 広さだけ
ju = smf.ols("家賃 ~ 広さ + 築年数 + 駅距離", data=df).fit()  # 3要因
print(f"単回帰 広さの係数: {tan.params['広さ']:.3f}")
print(f"重回帰 広さの係数: {ju.params['広さ']:.3f}")
print(f"重回帰 築年数の係数: {ju.params['築年数']:.3f}")
print(f"重回帰 駅距離の係数: {ju.params['駅距離']:.3f}")

print(ju.summary().tables[1])

# ===== LESSON 51 文字の情報を数式に入れる　ダミー変数 =====

import numpy as np
import statsmodels.formula.api as smf

rng2 = np.random.default_rng(7)
muki = rng2.choice(["北", "南"], n)                     # 各物件の向き
df["向き"] = muki
# 元の家賃は残し、向きの効果(南は+5万)を足した列を新しく作る
df["家賃_向き込み"] = df["家賃"] + np.where(muki == "南", 5, 0)

dm = smf.ols("家賃_向き込み ~ 広さ + C(向き)", data=df).fit()
for name, value in dm.params.items():
    print(f"{name}: {value:.2f}")

rng3 = np.random.default_rng(8)
df["方角"] = rng3.choice(["東", "南", "北"], n)      # 3種類
dm3 = smf.ols("家賃 ~ 広さ + C(方角)", data=df).fit()
for name in dm3.params.index:
    if "方角" in name:
        print(name)

# ===== LESSON 52 似た者どうしは入れない　多重共線性とVIF =====

import numpy as np
import statsmodels.formula.api as smf

df["畳数"] = df["広さ"] / 1.65 + rng.normal(0, 0.3, n)   # ほぼ広さと同じ

mc = smf.ols("家賃 ~ 広さ + 畳数 + 築年数 + 駅距離", data=df).fit()
print(f"広さの係数: {mc.params['広さ']:.2f}")
print(f"畳数の係数: {mc.params['畳数']:.2f}")

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df[["広さ", "畳数", "築年数", "駅距離"]]
Xc = sm.add_constant(X)                       # 切片ぶんの列を足す
for i, name in enumerate(Xc.columns):
    if name != "const":
        vif = variance_inflation_factor(Xc.values, i)
        print(f"{name} のVIF: {vif:.1f}")

naosi = smf.ols("家賃 ~ 広さ + 築年数 + 駅距離", data=df).fit()
print(f"畳数を消した後の広さの係数: {naosi.params['広さ']:.2f}")

# ===== LESSON 53 増やせば当たるは罠　過学習と調整済みR² =====

import numpy as np
import statsmodels.formula.api as smf

df2 = df.copy()
for i in range(5):
    df2[f"デタラメ{i}"] = rng.normal(0, 1, n)     # 家賃と無関係な乱数

moto = smf.ols("家賃 ~ 広さ + 築年数 + 駅距離", data=df2).fit()
shiki = "家賃 ~ 広さ + 築年数 + 駅距離 + " + \
        " + ".join(f"デタラメ{i}" for i in range(5))
fueta = smf.ols(shiki, data=df2).fit()

print(f"元のモデル      R²={moto.rsquared:.4f}  "
      f"調整済みR²={moto.rsquared_adj:.4f}")
print(f"デタラメ5個追加  R²={fueta.rsquared:.4f}  "
      f"調整済みR²={fueta.rsquared_adj:.4f}")

cols = ["家賃", "広さ", "築年数", "駅距離"]
z = (df[cols] - df[cols].mean()) / df[cols].std()   # 全部を標準化
zmod = smf.ols("家賃 ~ 広さ + 築年数 + 駅距離", data=z).fit()
print(f"標準化した係数  広さ: {zmod.params['広さ']:.3f}")
print(f"標準化した係数  築年数: {zmod.params['築年数']:.3f}")
print(f"標準化した係数  駅距離: {zmod.params['駅距離']:.3f}")

