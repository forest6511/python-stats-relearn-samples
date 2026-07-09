"""第7章 カイ二乗検定
『Pythonで学び直す統計・確率』サンプルコード。
Google Colab にこのファイルの内容を貼り付けて実行できます。
収録 LESSON: LESSON 37, LESSON 38, LESSON 39, LESSON 40
"""

# --- 共通の準備(序章のフォント設定セル相当) ---
# Colab では 1 回だけ実行すれば OK。日本語ラベルの豆腐(□)化を防ぐ。
!pip install -q matplotlib-fontja
import matplotlib_fontja  # noqa: F401
import matplotlib.pyplot as plt


# ===== LESSON 37 男女で好みは違う? 「関係がない世界」を作る =====

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

sex = np.array([0] * 120 + [1] * 130)     # 0=男性, 1=女性
like = np.array([1] * 136 + [0] * 114)    # 1=好き(136人), 0=好きでない

def chi2_value(tab):                       # 表から「ズレの合計」を出す
    row = tab.sum(axis=1)                  # 各行の合計(男女別の人数)
    col = tab.sum(axis=0)                  # 各列の合計(好き/好きでない)
    exp = np.outer(row, col) / tab.sum()   # 期待度数 = 按分
    return ((tab - exp) ** 2 / exp).sum()  # (観測-期待)^2 / 期待 の合計

kansoku = np.array([[84, 36], [52, 78]])   # 観測したクロス表
chi_obs = chi2_value(kansoku)

chi_null = []                              # 関係がない世界で出たズレの合計
for _ in range(10000):
    shuf = rng.permutation(like)           # 好き/好きでないをシャッフル
    tab = pd.crosstab(sex, shuf).values    # 配り直してクロス表を作る
    chi_null.append(chi2_value(tab))
chi_null = np.array(chi_null)

p = np.mean(chi_null >= chi_obs)
print(f"観測のズレの合計(カイ二乗値): {chi_obs:.1f}")
print(f"関係がない世界で、これ以上のズレが出た割合: {p * 100:.2f}%")

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 4))
plt.hist(chi_null, bins=np.linspace(0, 25, 50), color="0.8",
         edgecolor="black")
plt.axvline(chi_obs, color="black", lw=2, linestyle="--")
plt.xlabel("関係がない世界でのズレの合計(カイ二乗値)")
plt.ylabel("回数")
plt.show()

kansoku2 = np.array([[74, 46], [62, 68]])   # 男女差を小さくした表
chi_obs2 = chi2_value(kansoku2)
p2 = np.mean(chi_null >= chi_obs2)
print(f"ズレの合計: {chi_obs2:.1f}  珍しさ: {p2 * 100:.1f}%")

# ===== LESSON 38 1行で答え合わせ　独立性の検定 =====

from scipy import stats

kansoku = np.array([[84, 36], [52, 78]])
res = stats.chi2_contingency(kansoku, correction=False)

print(f"カイ二乗値(statistic): {res.statistic:.1f}")
print(f"p値(pvalue): {res.pvalue:.6f}")
print(f"自由度(dof): {res.dof}")
print("期待度数(expected_freq):")
print(res.expected_freq.round(1))

res_default = stats.chi2_contingency(kansoku)   # correction は既定の True
print(f"補正あり: カイ二乗値 {res_default.statistic:.1f}")
print(f"補正なし: カイ二乗値 {res.statistic:.1f}")

# ===== LESSON 39 サイコロは公平か　適合度の検定 =====

from scipy import stats

# サイコロを600回振って、各目が出た回数
saikoro = np.array([92, 105, 98, 110, 95, 100])
res_dice = stats.chisquare(saikoro)
print(f"サイコロ  カイ二乗値: {res_dice.statistic:.1f}  "
      f"p値: {res_dice.pvalue:.3f}")

# 曜日別の問い合わせ件数(月-金)
youbi = np.array([210, 150, 160, 155, 225])
res_day = stats.chisquare(youbi)
print(f"問い合わせ  カイ二乗値: {res_day.statistic:.1f}  "
      f"p値: {res_day.pvalue:.6f}")

# 100人が来ると想定。平日60・休日40の想定に対し、実際は平日50・休日50だった
kansoku = np.array([50, 50])
kitai = np.array([60, 40])
res = stats.chisquare(kansoku, f_exp=kitai)
print(f"カイ二乗値: {res.statistic:.2f}  p値: {res.pvalue:.3f}")

# ===== LESSON 40 落とし穴と使い分け　この道具をいつ使うか =====

from scipy import stats

# 全部で20人しかいない小さな表
small = np.array([[8, 2], [3, 7]])
row = small.sum(axis=1)
col = small.sum(axis=0)
exp = np.outer(row, col) / small.sum()
print(f"期待度数の最小: {exp.min():.1f}  (5未満なら注意)")

res_chi = stats.chi2_contingency(small)
odds, p_fisher = stats.fisher_exact(small)
print(f"カイ二乗検定 p値: {res_chi.pvalue:.3f}")
print(f"フィッシャー正確検定 p値: {p_fisher:.3f}")

base = np.array([[84, 36], [52, 78]])
for bai in [1, 10]:
    tab = base * bai
    res = stats.chi2_contingency(tab, correction=False)
    n = tab.sum()
    v = np.sqrt(res.statistic / (n * (min(tab.shape) - 1)))
    print(f"{bai}倍データ  p値: {res.pvalue:.1e}  効果量V: {v:.3f}")

