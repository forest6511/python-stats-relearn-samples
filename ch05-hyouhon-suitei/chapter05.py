"""第5章 標本・大数の法則・中心極限定理
『Pythonで学び直す統計・確率』サンプルコード。
Google Colab にこのファイルの内容を貼り付けて実行できます。
収録 LESSON: LESSON 23, LESSON 24, LESSON 25, LESSON 26, LESSON 27, LESSON 28, LESSON 29
"""

# --- 共通の準備(序章のフォント設定セル相当) ---
# Colab では 1 回だけ実行すれば OK。日本語ラベルの豆腐(□)化を防ぐ。
!pip install -q matplotlib-fontja
import matplotlib_fontja  # noqa: F401
import matplotlib.pyplot as plt


# ===== LESSON 23 1,000人の味見で、10万人の街が分かるのか =====

import numpy as np

rng = np.random.default_rng(42)

machi = np.zeros(100000, dtype=int)   # 10万人の街(0=不支持)
machi[:42000] = 1                     # 先頭の4万2,000人を支持(1)にする
machi = rng.permutation(machi)        # 街じゅうをよくかき混ぜる
print(f"街全体の支持率: {machi.mean() * 100:.1f}%")

hyohon = rng.choice(machi, size=1000, replace=False)   # くじ引きで1,000人選ぶ
print(f"標本1,000人の支持率: {hyohon.mean() * 100:.1f}%")

for kai in range(5):
    hyohon = rng.choice(machi, size=1000, replace=False)
    print(f"{kai + 1}回目の1,000人: 支持率 {hyohon.mean() * 100:.1f}%")

rng = np.random.default_rng(42)

nenrei = rng.integers(20, 80, size=100000)          # 20〜79歳の街
shijiritsu = 0.62 - 0.007 * (nenrei - 20)           # 若い人ほど支持しやすい
machi2 = (rng.random(100000) < shijiritsu).astype(int)
meibo = machi2[np.argsort(nenrei)]                  # 若い順に並べた名簿
print(f"街全体の支持率: {machi2.mean() * 100:.1f}%")
print(f"名簿の先頭1,000人だけ: {meibo[:1000].mean() * 100:.1f}%")

# ===== LESSON 24 増やせば増やすほど正確になる。ただし、ゆっくり =====

import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

plt.figure(figsize=(7, 4))
for sen in ["-", "--", ":"]:
    kotae = rng.choice(machi, size=10000, replace=False)   # 1万人に順に聞く
    keika = np.cumsum(kotae) / np.arange(1, 10001) * 100   # n人目までの支持率
    plt.plot(keika, color="black", linestyle=sen, lw=1)
plt.axhline(42, color="0.5")
plt.xscale("log")
plt.xlabel("聞いた人数")
plt.ylabel("そこまでの支持率(%)")
plt.show()

rng = np.random.default_rng(42)
for n in [100, 400, 1600]:
    shiji = [rng.choice(machi, size=n, replace=False).mean()
             for _ in range(1000)]
    print(f"{n:>4}人調査のブレ幅: {np.std(shiji) * 100:.2f}ポイント")

# ===== LESSON 25 調査のたびに答えは違う。でも、ブレ方には形がある =====

rng = np.random.default_rng(42)

tsuwa = rng.exponential(10, size=100000)   # 10万件の通話時間(分)
print(f"母集団の平均: {tsuwa.mean():.2f}分 / 標準偏差: {tsuwa.std():.2f}分")

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
axes[0].hist(tsuwa, bins=50, color="0.8", edgecolor="black")
axes[0].set_title("母集団: 通話時間そのもの", fontsize=10)
for ax, n in zip(axes[1:], [5, 30]):
    heikin = rng.choice(tsuwa, size=(10000, n)).mean(axis=1)
    ax.hist(heikin, bins=50, color="0.8", edgecolor="black")
    ax.set_title(f"{n}件の標本平均を1万回", fontsize=10)
plt.tight_layout()
plt.show()

rng = np.random.default_rng(42)
heikin30 = rng.choice(tsuwa, size=(10000, 30)).mean(axis=1)
print(f"標本平均のブレ幅(実測): {np.std(heikin30):.3f}分")
print(f"母集団の標準偏差 ÷ √30: {tsuwa.std() / np.sqrt(30):.3f}分")

rng = np.random.default_rng(42)

karui = rng.normal(5, 1, size=50000)        # 軽い手続き: 5分前後
fukuzatsu = rng.normal(20, 2, size=50000)   # 複雑な手続き: 20分前後
futayama = np.concatenate([karui, fukuzatsu])
heikin30 = rng.choice(futayama, size=(10000, 30)).mean(axis=1)
print(f"母集団の平均: {futayama.mean():.2f}分")
print(f"標本平均の中心: {heikin30.mean():.2f}分 / ブレ幅: {np.std(heikin30):.3f}分")
print(f"σ ÷ √30 の予言: {futayama.std() / np.sqrt(30):.3f}分")

tsuwa5 = [12, 3, 8, 25, 2]          # 通話時間5件(分)
goukei = 0                          # 総和の入れ物
for x in tsuwa5:                    # i を 1 から n まで動かす、に対応
    goukei = goukei + x             # x_i を足しこむ = Σ の仕事
print(goukei / len(tsuwa5))         # 最後に n で割る = 標本平均

# ===== LESSON 26 なぜ n ではなく n−1 で割るのか =====

rng = np.random.default_rng(42)

waru_n, waru_n1 = [], []
for _ in range(10000):
    hyohon = rng.normal(50, 10, size=5)        # 母分散100の世界から5件
    waru_n.append(np.var(hyohon))              # n で割る(第2章のやり方)
    waru_n1.append(np.var(hyohon, ddof=1))     # n-1 で割る
print(f"本当の母分散: 100")
print(f"n で割った分散の平均:   {np.mean(waru_n):.1f}")
print(f"n-1 で割った分散の平均: {np.mean(waru_n1):.1f}")

rng = np.random.default_rng(42)
for n in [5, 30, 100]:
    b = [np.var(rng.normal(50, 10, size=n)) for _ in range(10000)]
    print(f"n = {n:>3}: n で割った分散の平均 {np.mean(b):.1f}")

# ===== LESSON 27 点で答えると外れる。「まず外さない幅」で答える =====

rng = np.random.default_rng(42)

sigma = 6                                   # 母標準偏差(今回は既知とする)
hyohon = rng.normal(171, sigma, size=30)    # 30人の身長を測る
xbar = hyohon.mean()
haba = 1.96 * sigma / np.sqrt(30)
print(f"標本平均: {xbar:.1f}cm")
print(f"95%信頼区間: {xbar - haba:.1f}cm 〜 {xbar + haba:.1f}cm")

rng = np.random.default_rng(42)

atari = 0
plt.figure(figsize=(7, 4.5))
for kai in range(100):
    xbar = rng.normal(171, 6, size=30).mean()
    fukumu = xbar - haba <= 171 <= xbar + haba
    atari += fukumu
    plt.plot([kai, kai], [xbar - haba, xbar + haba],
             color="0.65" if fukumu else "black", lw=1.2 if fukumu else 2.5)
plt.axhline(171, color="black", linestyle="--")
plt.xlabel("何回目の調査か")
plt.ylabel("身長(cm)")
plt.show()
print(f"母平均171cmを捕まえた区間: {atari}本 / 100本")

rng = np.random.default_rng(42)
for z, suijun in [(1.96, 95), (1.28, 80)]:
    xbar = rng.normal(171, 6, size=(1000, 30)).mean(axis=1)
    ataru = np.mean(np.abs(xbar - 171) <= z * 6 / np.sqrt(30))
    print(f"{suijun}%の輪(幅 ±{z * 6 / np.sqrt(30):.1f}cm): "
          f"捕まえた割合 {ataru * 100:.1f}%")

# ===== LESSON 28 ニュースの「誤差±3ポイント」を自分で出す =====

rng = np.random.default_rng(42)

shin = 0.42                                        # 真の支持率(神の視点)
ritsu = rng.binomial(1000, shin, size=10000) / 1000
bure = np.std(ritsu)
print(f"1,000人調査の支持率のブレ幅: {bure * 100:.2f}ポイント")
print(f"±{1.96 * bure * 100:.1f}ポイントに入った調査: "
      f"{np.mean(np.abs(ritsu - shin) <= 1.96 * bure) * 100:.1f}%")

rng = np.random.default_rng(42)
for shin in [0.05, 0.30, 0.50]:
    ritsu = rng.binomial(1000, shin, size=10000) / 1000
    print(f"真の支持率{shin * 100:>3.0f}%: ブレ幅 {np.std(ritsu) * 100:.2f}ポイント")

# ===== LESSON 29 何人に聞けばいいかは、調査の前に決められる =====

mokuhyo = 0.03                            # ほしい誤差 ±3ポイント
n = 1.96**2 * 0.5 * 0.5 / mokuhyo**2
print(f"必要な人数: {n:.1f}人")

rng = np.random.default_rng(42)
ritsu = rng.binomial(1068, 0.42, size=10000) / 1068
print(f"±3ポイントに入った調査: "
      f"{np.mean(np.abs(ritsu - 0.42) <= 0.03) * 100:.1f}%")

for gosa in [0.03, 0.02, 0.01]:
    n = 1.96**2 * 0.25 / gosa**2
    print(f"±{gosa * 100:.0f}ポイントなら: {n:,.0f}人")

n = 500          # 自分のアンケートの回答数に書き換える
hai = 0.62       # 「はい」の割合に書き換える
haba = 1.96 * np.sqrt(hai * (1 - hai) / n)
print(f"95%信頼区間: {(hai - haba) * 100:.1f}% 〜 {(hai + haba) * 100:.1f}%")

