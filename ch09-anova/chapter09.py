"""第9章 分散分析(ANOVA)
『Pythonで学び直す統計・確率』サンプルコード。
Google Colab にこのファイルの内容を貼り付けて実行できます。
収録 LESSON: LESSON 46, LESSON 47, LESSON 48, LESSON 49
"""

# --- 共通の準備(序章のフォント設定セル相当) ---
# Colab では 1 回だけ実行すれば OK。日本語ラベルの豆腐(□)化を防ぐ。
!pip install -q matplotlib-fontja
import matplotlib_fontja  # noqa: F401
import matplotlib.pyplot as plt


# ===== LESSON 46 t検定を3回やってはいけない　多重検定の罠 =====

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

madeguri = 0                      # 3回中どれか有意だったセット数
for _ in range(10000):
    a = rng.normal(50, 10, 30)    # 3店とも同じ設定(差がない)
    b = rng.normal(50, 10, 30)
    c = rng.normal(50, 10, 30)
    p_ab = stats.ttest_ind(a, b).pvalue
    p_ac = stats.ttest_ind(a, c).pvalue
    p_bc = stats.ttest_ind(b, c).pvalue
    if min(p_ab, p_ac, p_bc) < 0.05:   # どれか1つでも有意なら
        madeguri += 1

print(f"3回検定でどれかが有意になった割合: {madeguri / 10000 * 100:.1f}%")

from itertools import combinations

for mise in [3, 5]:
    madeguri = 0
    for _ in range(10000):
        groups = [rng.normal(50, 10, 30) for _ in range(mise)]
        ps = [stats.ttest_ind(g1, g2).pvalue
              for g1, g2 in combinations(groups, 2)]
        if min(ps) < 0.05:
            madeguri += 1
    kaisuu = mise * (mise - 1) // 2
    print(f"{mise}店({kaisuu}回検定): まぐれ有意 {madeguri / 10000 * 100:.1f}%")

# ===== LESSON 47 ばらつきで差を見抜く　一元配置分散分析 =====

import numpy as np
from scipy import stats

rng = np.random.default_rng(1)
mise1 = rng.normal(100, 15, 20)      # 平均100の店
mise2 = rng.normal(108, 15, 20)      # 平均108の店
mise3 = rng.normal(120, 15, 20)      # 平均120の店

res = stats.f_oneway(mise1, mise2, mise3)
print(f"3店の分散分析  F比 = {res.statistic:.2f}  p値 = {res.pvalue:.4f}")

# 比較: 差がない3店だと
n1 = rng.normal(100, 15, 20)
n2 = rng.normal(100, 15, 20)
n3 = rng.normal(100, 15, 20)
res_nashi = stats.f_oneway(n1, n2, n3)
print(f"差がない3店  F比 = {res_nashi.statistic:.2f}  "
      f"p値 = {res_nashi.pvalue:.3f}")

import matplotlib.pyplot as plt

pooled = np.concatenate([mise1, mise2, mise3])   # 3店を混ぜる
F_null = []
for _ in range(10000):
    sh = rng.permutation(pooled)                 # 混ぜて3群に配り直す
    F_null.append(stats.f_oneway(sh[:20], sh[20:40], sh[40:]).statistic)
F_null = np.array(F_null)

plt.figure(figsize=(7, 4))
plt.hist(F_null, bins=np.linspace(0, 10, 50), color="0.8", edgecolor="black")
plt.axvline(res.statistic, color="black", lw=2, linestyle="--")
plt.xlabel("差がない世界でのF比")
plt.ylabel("回数")
plt.show()

rng2 = np.random.default_rng(1)
m1 = rng2.normal(100, 30, 20)     # ばらつきだけ2倍(15→30)
m2 = rng2.normal(108, 30, 20)
m3 = rng2.normal(120, 30, 20)
res_baraaki = stats.f_oneway(m1, m2, m3)
print(f"ばらつき2倍  F比 = {res_baraaki.statistic:.2f}  "
      f"p値 = {res_baraaki.pvalue:.3f}")

# ===== LESSON 48 どの組に差がある?　多重比較（Tukey） =====

import numpy as np
from scipy import stats

rng = np.random.default_rng(1)
mise1 = rng.normal(100, 15, 20)
mise2 = rng.normal(108, 15, 20)
mise3 = rng.normal(120, 15, 20)

res = stats.tukey_hsd(mise1, mise2, mise3)
print("組み合わせごとの p値(0が店1, 1が店2, 2が店3):")
print(res.pvalue.round(4))

ci = res.confidence_interval()
print(f"店1と店3の差の95%信頼区間: "
      f"{ci.low[0, 2]:.1f} 〜 {ci.high[0, 2]:.1f}")

print(f"店1と店2(有意でない)の差の95%信頼区間: "
      f"{ci.low[0, 1]:.1f} 〜 {ci.high[0, 1]:.1f}")

# ===== LESSON 49 使う前の前提チェック　等分散・正規性と効果量 =====

import numpy as np
from scipy import stats

rng = np.random.default_rng(1)
mise1 = rng.normal(100, 15, 20)
mise2 = rng.normal(108, 15, 20)
mise3 = rng.normal(120, 15, 20)

lev = stats.levene(mise1, mise2, mise3)
print(f"等分散の検定(Levene) p値 = {lev.pvalue:.3f}")

kw = stats.kruskal(mise1, mise2, mise3)
print(f"ノンパラメトリック(Kruskal) p値 = {kw.pvalue:.4f}")

zentai = np.concatenate([mise1, mise2, mise3])
grand = zentai.mean()                                    # 全体の平均
gunkan = sum(len(g) * (g.mean() - grand) ** 2
             for g in [mise1, mise2, mise3])             # 群間の変動
zentai_hendou = ((zentai - grand) ** 2).sum()            # 全体の変動
eta2 = gunkan / zentai_hendou
print(f"効果量 η^2 = {eta2:.3f}")

rng2 = np.random.default_rng(1)
h1 = rng2.normal(100, 5, 20)      # ばらつき小
h2 = rng2.normal(108, 5, 20)      # ばらつき小
h3 = rng2.normal(120, 40, 20)     # 店3だけばらつき特大
lev2 = stats.levene(h1, h2, h3)
print(f"店3だけばらつき特大  Levene p値 = {lev2.pvalue:.4f}")

