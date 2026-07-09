"""第13章 総合演習
『Pythonで学び直す統計・確率』サンプルコード。
Google Colab にこのファイルの内容を貼り付けて実行できます。
収録 LESSON: LESSON 60
"""

# --- 共通の準備(序章のフォント設定セル相当) ---
# Colab では 1 回だけ実行すれば OK。日本語ラベルの豆腐(□)化を防ぐ。
!pip install -q matplotlib-fontja
import matplotlib_fontja  # noqa: F401
import matplotlib.pyplot as plt


# ===== LESSON 60 世帯の食費データを分析しきる =====

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 300
jinzu = rng.integers(1, 6, n)             # 世帯人数(1〜5人)
nenshu = rng.uniform(300, 900, n)         # 年収(万円)
toshi = rng.choice([0, 1], n)             # 0=地方, 1=都市
shokuhi = 15 * jinzu + 0.02 * nenshu + 8 * toshi + 10 + rng.normal(0, 6, n)
df = pd.DataFrame({"食費": shokuhi.round(1), "世帯人数": jinzu,
                   "年収": nenshu.round(0), "都市": toshi})

print(f"件数: {len(df)}   欠損: {df.isna().sum().sum()}")
print(f"食費 平均: {df['食費'].mean():.1f}千円")
print(f"食費 中央値: {df['食費'].median():.1f}千円")
print(f"食費 標準偏差: {df['食費'].std():.1f}千円")

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.hist(df["食費"], bins=20, color="0.8", edgecolor="black")
plt.axvline(df["食費"].mean(), color="black", lw=2, label="平均")
plt.xlabel("月間食費(千円)")
plt.ylabel("世帯数")
plt.legend()
plt.show()

from scipy import stats

toshi_d = df[df["都市"] == 1]["食費"]      # 都市の食費
chihou = df[df["都市"] == 0]["食費"]       # 地方の食費
kekka = stats.ttest_ind(toshi_d, chihou, equal_var=False)
ci = kekka.confidence_interval()

print(f"都市の平均: {toshi_d.mean():.1f}千円")
print(f"地方の平均: {chihou.mean():.1f}千円")
print(f"p値: {kekka.pvalue:.4f}")
print(f"差の95%信頼区間: {ci.low:.1f} 〜 {ci.high:.1f}千円")

sukunai = df[df["世帯人数"] <= 2]["食費"]
chuu = df[df["世帯人数"] == 3]["食費"]
ooi = df[df["世帯人数"] >= 4]["食費"]
f = stats.f_oneway(sukunai, chuu, ooi)
print(f"分散分析  F比: {f.statistic:.1f}  p値: {f.pvalue:.2e}")

r = np.corrcoef(df["年収"], df["食費"])[0, 1]
katamuki, seppen = np.polyfit(df["年収"], df["食費"], 1)
print(f"年収と食費の相関: r = {r:.3f}")
print(f"回帰の傾き: {katamuki:.4f}")

plt.figure(figsize=(6, 4))
plt.scatter(df["年収"], df["食費"], color="0.55", s=15)
xs = np.array([300, 900])
plt.plot(xs, katamuki * xs + seppen, color="black", lw=2)
plt.xlabel("年収(万円)")
plt.ylabel("月間食費(千円)")
plt.show()

import statsmodels.formula.api as smf

model = smf.ols("食費 ~ 世帯人数 + 年収 + C(都市)", data=df).fit()
print(f"決定係数 R²: {model.rsquared:.3f}")
print(f"世帯人数の係数: {model.params['世帯人数']:.2f}")
print(f"年収の係数: {model.params['年収']:.3f}")
print(f"都市の係数: {model.params['C(都市)[T.1]']:.2f}")

plt.figure(figsize=(6, 4))
data = [chihou, toshi_d]
bp = plt.boxplot(data, patch_artist=True)
for hako in bp["boxes"]:
    hako.set_facecolor("0.85")
plt.xticks([1, 2], ["地方", "都市"])
plt.ylabel("月間食費(千円)")
plt.show()

