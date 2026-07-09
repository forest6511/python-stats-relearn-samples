"""第4章 分布
『Pythonで学び直す統計・確率』サンプルコード。
Google Colab にこのファイルの内容を貼り付けて実行できます。
収録 LESSON: LESSON 17, LESSON 18, LESSON 19, LESSON 20, LESSON 21, LESSON 22
"""

# --- 共通の準備(序章のフォント設定セル相当) ---
# Colab では 1 回だけ実行すれば OK。日本語ラベルの豆腐(□)化を防ぐ。
!pip install -q matplotlib-fontja
import matplotlib_fontja  # noqa: F401
import matplotlib.pyplot as plt


# ===== LESSON 17 コイン10回、表は何回出るか予想できるか =====

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

omote = rng.integers(0, 2, size=(10000, 10))   # コイン10回×1万セット
kaisuu = omote.sum(axis=1)                     # セットごとの表の回数

plt.figure(figsize=(7, 4))
plt.hist(kaisuu, bins=np.arange(-0.5, 11.5, 1), color="0.8",
         edgecolor="black")
plt.xlabel("10回中、表が出た回数")
plt.ylabel("セット数")
plt.show()

from scipy import stats

for k in [0, 3, 5]:
    riron = stats.binom.pmf(k, 10, 0.5) * 10000
    jissai = np.sum(kaisuu == k)
    print(f"表{k}回: 実験 {jissai}セット / 理論 {riron:.0f}セット")

rng = np.random.default_rng(42)
seikou = (rng.random((10000, 10)) < 0.2).sum(axis=1)   # 成功率2割で10本
for k in [0, 1, 2, 3, 5]:
    print(f"成功{k}回: {np.sum(seikou == k):>4}セット")

# ===== LESSON 18 パチンコ玉が教えてくれる、釣鐘型の正体 =====

rng = np.random.default_rng(42)

fig, axes = plt.subplots(2, 2, figsize=(9, 6))
for ax, n in zip(axes.ravel(), [1, 3, 10, 100]):
    goukei = rng.integers(0, 2, size=(10000, n)).sum(axis=1)
    ax.hist(goukei, bins=np.arange(-0.5, n + 1.5, 1), color="0.8",
            edgecolor="black")
    ax.set_title(f"偶然を {n} 個足し合わせる", fontsize=11)
plt.tight_layout()
plt.show()

rng = np.random.default_rng(42)
saikoro = rng.integers(1, 7, size=(10000, 100)).sum(axis=1)
print(f"平均: {saikoro.mean():.1f} / 標準偏差: {saikoro.std():.1f}")

from scipy import stats

rng = np.random.default_rng(42)
goukei = rng.integers(0, 2, size=(10000, 100)).sum(axis=1)

x = np.arange(30, 71)
riron = stats.norm.pdf(x, goukei.mean(), goukei.std()) * 10000

plt.figure(figsize=(7, 4))
plt.hist(goukei, bins=np.arange(29.5, 71.5, 1), color="0.8",
         edgecolor="black")
plt.plot(x, riron, color="black", lw=2)
plt.xlabel("表の回数(コイン100回)")
plt.ylabel("セット数")
plt.show()

# ===== LESSON 19 正規分布の設計図は、たった2つの数字 =====

from scipy import stats

x = np.linspace(140, 200, 200)
plt.figure(figsize=(7, 4))
for m, s, sen in [(171, 6, "-"), (158, 5, "--"), (171, 12, ":")]:
    plt.plot(x, stats.norm.pdf(x, m, s), color="black", linestyle=sen,
             label=f"平均{m}cm・標準偏差{s}cm")
plt.legend()
plt.xlabel("身長(cm)")
plt.ylabel("起こりやすさの目安")
plt.show()

rng = np.random.default_rng(42)
shincho = rng.normal(171, 6, size=10000)   # 正規分布の世界の1万人
for haba in [1, 2, 3]:
    naka = np.mean(np.abs(shincho - 171) < 6 * haba)
    print(f"平均±標準偏差{haba}つ分: {naka * 100:.1f}%")

print(f"180cm以上の人: {np.mean(shincho >= 180) * 100:.1f}%")

# ===== LESSON 20 偏差値の完全な種明かし: 標準化 =====

z_sugaku = (90 - 85) / 10
z_eigo = (75 - 55) / 10
print(f"数学90点: z = {z_sugaku}")
print(f"英語75点: z = {z_eigo}")

from scipy import stats

print(f"z=2.0 より下: {stats.norm.cdf(2.0) * 100:.1f}%")
print(f"z=2.0 より上: {(1 - stats.norm.cdf(2.0)) * 100:.1f}%")

for hensachi in [40, 50, 60, 70, 80]:
    z = (hensachi - 50) / 10
    ue = (1 - stats.norm.cdf(z)) * 100
    print(f"偏差値{hensachi}: 上位 {ue:.1f}%")

# ===== LESSON 21 ガタガタの棒が、曲線に溶けるとき =====

rng = np.random.default_rng(42)

setai = rng.binomial(1000, 0.2, size=10000)   # 視聴率20%を1,000世帯で測る
x = np.arange(150, 251)
riron = stats.norm.pdf(x, 200, np.sqrt(1000 * 0.2 * 0.8)) * 10000

plt.figure(figsize=(7, 4))
plt.hist(setai, bins=np.arange(149.5, 251.5, 1), color="0.8",
         edgecolor="black")
plt.plot(x, riron, color="black", lw=2)
plt.xlabel("1,000世帯のうち見ていた世帯数")
plt.ylabel("調査のセット数")
plt.show()

rng = np.random.default_rng(42)
mare = rng.binomial(20, 0.05, size=10000)
for k in [0, 1, 2, 3]:
    print(f"{k}回: {np.sum(mare == k):>4}セット")

# ===== LESSON 22 世界は釣鐘だけではない =====

rng = np.random.default_rng(42)
matiawase = rng.random(10000) * 60
madoguchi = rng.exponential(10, 10000)
nenshu = rng.lognormal(6.2, 0.55, 10000)

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
mei = ["いつでも同じ(一様分布)", "待ち時間(指数分布)", "年収の形(対数正規分布)"]
for ax, d, t in zip(axes, [matiawase, madoguchi, nenshu], mei):
    ax.hist(d, bins=40, color="0.8", edgecolor="black")
    ax.set_title(t, fontsize=10)
plt.tight_layout()
plt.show()

taisuu = np.log(nenshu)
print(f"元の年収:  平均 {nenshu.mean():.0f}万円 / 中央値 {np.median(nenshu):.0f}万円")
print(f"logの世界: 平均 {taisuu.mean():.2f} / 中央値 {np.median(taisuu):.2f}")

from scipy import stats

data = np.array([6.5, 7.0, 5.5, 8.0, 6.0, 7.5, 6.5, 5.0, 9.0, 6.5])
m, s = np.mean(data), np.std(data)

plt.figure(figsize=(7, 4))
plt.hist(data, bins=6, density=True, color="0.8", edgecolor="black")
x = np.linspace(m - 3 * s, m + 3 * s, 100)
plt.plot(x, stats.norm.pdf(x, m, s), color="black", lw=2)
plt.xlabel("睡眠時間(時間)")
plt.show()

