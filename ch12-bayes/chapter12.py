"""第12章 ベイズ統計入門
『Pythonで学び直す統計・確率』サンプルコード。
Google Colab にこのファイルの内容を貼り付けて実行できます。
収録 LESSON: LESSON 57, LESSON 58, LESSON 59
"""

# --- 共通の準備(序章のフォント設定セル相当) ---
# Colab では 1 回だけ実行すれば OK。日本語ラベルの豆腐(□)化を防ぐ。
!pip install -q matplotlib-fontja
import matplotlib_fontja  # noqa: F401
import matplotlib.pyplot as plt


# ===== LESSON 57 信じ具合を更新する　ベイズの定理再訪 =====

N = 100000                       # 全体の人数
byoki = int(N * 0.01)            # 事前: 有病率1%(病気の人)
kenkou = N - byoki               # 病気でない人
yousei_byoki = int(byoki * 0.99)      # 病気で陽性(感度99%)
yousei_kenkou = int(kenkou * 0.01)    # 健康なのに陽性(偽陽性1%)

ppv = yousei_byoki / (yousei_byoki + yousei_kenkou)
print(f"病気で陽性の人: {yousei_byoki}人")
print(f"健康なのに陽性の人: {yousei_kenkou}人")
print(f"陽性のうち本当に病気(事後確率): {ppv:.3f}")

for yuubyou in [0.01, 0.10]:
    byoki = int(N * yuubyou)
    kenkou = N - byoki
    yb = int(byoki * 0.99)
    yk = int(kenkou * 0.01)
    ppv = yb / (yb + yk)
    print(f"有病率{yuubyou:.0%}のとき、陽性の事後確率: {ppv:.3f}")

# ===== LESSON 58 分布で答える　グリッド近似で事後分布を作る =====

import numpy as np
from scipy import stats

grid = np.linspace(0, 1, 101)          # 表率の候補を0〜1で101個
prior = np.ones_like(grid)             # 事前は真っ平ら(何も知らない)
prior = prior / prior.sum()

# 10回投げて7回表だった、というデータの尤もらしさ
likelihood = stats.binom.pmf(7, 10, grid)
post = prior * likelihood              # 事前 × 尤度
post = post / post.sum()               # 合計1になるようそろえる

heikin = (grid * post).sum()           # 事後分布の平均
yama = grid[np.argmax(post)]           # 事後分布の山の位置
print(f"10回中7回表のあと、事後平均: {heikin:.3f}")
print(f"事後分布の山(最も濃い表率): {yama:.3f}")

def jigo(k, n):                        # k回表/n回投げ の事後分布を返す
    lik = stats.binom.pmf(k, n, grid)
    po = prior * lik
    return po / po.sum()

for k, n in [(7, 10), (70, 100)]:
    po = jigo(k, n)
    heikin = (grid * po).sum()
    haba = np.sqrt(((grid - heikin) ** 2 * po).sum())   # 分布の幅
    print(f"{n}回中{k}回表: 事後平均{heikin:.3f}  幅{haba:.3f}")

# ===== LESSON 59 「Bが勝つ確率97%」と言える　ベイズA/Bテスト =====

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

# 事前を平ら(Beta(1,1))として、観測を足した事後分布
postA = stats.beta(1 + 30, 1 + 1000 - 30)      # A: 30/1000
postB = stats.beta(1 + 35, 1 + 1000 - 35)      # B: 35/1000

# 両方の事後分布から10万回サンプリングして、Bが上回る割合を数える
sample_A = postA.rvs(100000, random_state=rng)
sample_B = postB.rvs(100000, random_state=rng)
print(f"BがAより良い確率: {(sample_B > sample_A).mean():.3f}")

p1, p2 = 30 / 1000, 35 / 1000
pooled = (30 + 35) / (1000 + 1000)
se = np.sqrt(pooled * (1 - pooled) * (1 / 1000 + 1 / 1000))
z = (p2 - p1) / se
p_value = 1 - stats.norm.cdf(z)           # Bのほうが高い方向の片側p値
print(f"頻度論の片側p値: {p_value:.3f}")

