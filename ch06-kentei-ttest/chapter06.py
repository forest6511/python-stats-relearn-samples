"""第6章 仮説検定とt検定
『Pythonで学び直す統計・確率』サンプルコード。
Google Colab にこのファイルの内容を貼り付けて実行できます。
収録 LESSON: LESSON 30, LESSON 31, LESSON 32, LESSON 33, LESSON 34, LESSON 35, LESSON 36
"""

# --- 共通の準備(序章のフォント設定セル相当) ---
# Colab では 1 回だけ実行すれば OK。日本語ラベルの豆腐(□)化を防ぐ。
!pip install -q matplotlib-fontja
import matplotlib_fontja  # noqa: F401
import matplotlib.pyplot as plt


# ===== LESSON 30 その差は本物か。「差がない世界」を作って数える =====

import numpy as np

rng = np.random.default_rng(42)

click = np.zeros(2000, dtype=int)   # 2,000人分のクリック記録(1=クリック)
click[:102] = 1                     # クリックしたのは全部で102人(40+62)

sa_nai = []                         # 差がない世界で出た「クリック数の差」
for _ in range(10000):
    mazeta = rng.permutation(click)                  # ぜんぶ混ぜて
    sa = mazeta[:1000].sum() - mazeta[1000:].sum()   # 配り直して差を数える
    sa_nai.append(sa)
sa_nai = np.array(sa_nai)

p = np.mean(np.abs(sa_nai) >= 22)   # 観測された差は 62 - 40 = 22人
print(f"差がない世界で、差22人以上が出た割合: {p * 100:.2f}%")

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 4))
plt.hist(sa_nai, bins=np.arange(-41, 43, 2), color="0.8", edgecolor="black")
plt.axvline(22, color="black", lw=2)
plt.axvline(-22, color="black", lw=2, linestyle="--")
plt.xlabel("差がない世界での、クリック数の差(人)")
plt.ylabel("回数")
plt.show()

for kansoku in [22, 10]:
    p = np.mean(np.abs(sa_nai) >= kansoku)
    print(f"差{kansoku}人以上が出る割合: {p * 100:.1f}%")

# ===== LESSON 31 どこからが「まぐれじゃない」? 5%という線引き =====

rng = np.random.default_rng(42)

a = rng.binomial(1000, 0.051, size=10000)   # どちらも同じボタン(5.1%)
b = rng.binomial(1000, 0.051, size=10000)
sa = np.abs(a - b)                          # 差がない世界での差(人)
for sen in [19, 25]:
    enzai = np.mean(sa >= sen) * 100
    print(f"『差{sen}人以上で差あり』と判定: 冤罪率 {enzai:.1f}%")

rng = np.random.default_rng(42)

kyu = rng.binomial(1000, 0.040, size=10000)    # 本当に差がある世界
shin = rng.binomial(1000, 0.062, size=10000)   # (4.0%と6.2%のボタン)
sa2 = np.abs(shin - kyu)
for sen in [19, 25]:
    minogashi = np.mean(sa2 < sen) * 100
    print(f"差{sen}人の線: 見逃し率 {minogashi:.1f}%")

# ===== LESSON 32 お弁当は250g入っているか。はじめてのt検定 =====

rng = np.random.default_rng(42)

bento = rng.normal(252, 4, size=30)   # 最近詰めた30個の実測値(g)
print(f"30個の平均: {bento.mean():.1f}g")
print(f"30個の標準偏差(n-1版): {np.std(bento, ddof=1):.1f}g")

from scipy import stats

kekka = stats.ttest_1samp(bento, 250)   # 基準値250gと比べる
print(kekka)

print(f"t = {kekka.statistic:.2f}")
print(f"p値 = {kekka.pvalue:.4f}")
print(f"自由度 = {kekka.df}")

kekka2 = stats.ttest_1samp(bento, 252)   # 基準値を252gに変えてみる
print(f"p値 = {kekka2.pvalue:.4f}")

# ===== LESSON 33 どちらの研修が効いたのか。2つの平均を比べる =====

rng = np.random.default_rng(42)

kyu = rng.normal(50, 8, size=30)    # 旧研修を受けた30人の処理件数
shin = rng.normal(55, 8, size=30)   # 新研修を受けた30人の処理件数
print(f"旧研修の平均: {kyu.mean():.1f}件 / 新研修の平均: {shin.mean():.1f}件")

kekka = stats.ttest_ind(shin, kyu, equal_var=False)
print(f"t = {kekka.statistic:.2f} / p値 = {kekka.pvalue:.4f}")

ci = kekka.confidence_interval()
print(f"差の95%信頼区間: {ci.low:.1f}件 〜 {ci.high:.1f}件")

rng = np.random.default_rng(42)

kyu2 = rng.normal(50, 8, size=20000)     # 差はわずか0.3件の世界で
shin2 = rng.normal(50.3, 8, size=20000)  # 2万人ずつ集めると
kekka2 = stats.ttest_ind(shin2, kyu2, equal_var=False)
print(f"平均の差: {shin2.mean() - kyu2.mean():.2f}件")
print(f"p値: {kekka2.pvalue:.4f}")

# ===== LESSON 34 同じ人の「前と後」は、特別あつかいができる =====

rng = np.random.default_rng(42)

mae = rng.normal(50, 10, size=10)        # 研修前の処理件数(10人)
ato = mae + rng.normal(3, 2, size=10)    # 同じ10人の研修後
print(f"平均の伸び: {(ato - mae).mean():.1f}件")
print(f"対応ありのp値: {stats.ttest_rel(ato, mae).pvalue:.5f}")
print(f"対応を無視したp値: {stats.ttest_ind(ato, mae, equal_var=False).pvalue:.4f}")

print(f"対応のあるt検定:   p = {stats.ttest_rel(ato, mae).pvalue:.6f}")
print(f"伸びの1標本t検定: p = {stats.ttest_1samp(ato - mae, 0).pvalue:.6f}")

# ===== LESSON 35 20回試せば、まぐれの「有意」が出てしまう =====

rng = np.random.default_rng(42)

hikkakari = 0
for _ in range(1000):                    # 1,000回の「改善プロジェクト」
    yuui = 0
    for _ in range(20):                  # 1回につき20個の指標を検定
        a = rng.normal(50, 8, size=30)   # 差がない世界のA群とB群
        b = rng.normal(50, 8, size=30)
        if stats.ttest_ind(a, b, equal_var=False).pvalue < 0.05:
            yuui += 1
    if yuui >= 1:
        hikkakari += 1
print(f"どれか1つは『有意』が出たプロジェクト: {hikkakari / 10:.1f}%")

rng = np.random.default_rng(42)

hikkakari = 0
for _ in range(1000):
    yuui = 0
    for _ in range(20):
        a = rng.normal(50, 8, size=30)
        b = rng.normal(50, 8, size=30)
        if stats.ttest_ind(a, b, equal_var=False).pvalue < 0.05 / 20:
            yuui += 1
    if yuui >= 1:
        hikkakari += 1
print(f"線を 0.05÷20 に締めたとき: {hikkakari / 10:.1f}%")

# ===== LESSON 36 差があるのに「有意差なし」。人数が足りないだけかも =====

rng = np.random.default_rng(42)
for n in [10, 30, 100]:
    mitsuketa = 0
    for _ in range(2000):
        kyu = rng.normal(50, 8, size=n)     # 本当は+5件の差がある世界
        shin = rng.normal(55, 8, size=n)
        if stats.ttest_ind(shin, kyu, equal_var=False).pvalue < 0.05:
            mitsuketa += 1
    print(f"n = {n:>3}: 差を見つけられた割合 {mitsuketa / 20:.1f}%")

rng = np.random.default_rng(42)
mitsuketa = 0
for _ in range(2000):
    a = rng.normal(50, 8, size=30)          # 差がない世界
    b = rng.normal(50, 8, size=30)
    if stats.ttest_ind(b, a, equal_var=False).pvalue < 0.05:
        mitsuketa += 1
print(f"差がない世界で『差あり』と言った割合: {mitsuketa / 20:.1f}%")

a = np.array([12.4, 15.1, 9.8, 13.0, 11.2, 14.6, 10.5, 12.9])   # A案8日分
b = np.array([14.2, 16.8, 13.5, 15.0, 12.1, 16.2, 14.8, 13.9])  # B案8日分
kekka = stats.ttest_ind(b, a, equal_var=False)
ci = kekka.confidence_interval()
print(f"平均の差: {b.mean() - a.mean():.2f}件")
print(f"p値: {kekka.pvalue:.3f}")
print(f"差の95%信頼区間: {ci.low:.2f}件 〜 {ci.high:.2f}件")

