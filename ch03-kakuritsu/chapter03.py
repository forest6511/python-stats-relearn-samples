"""第3章 確率
『Pythonで学び直す統計・確率』サンプルコード。
Google Colab にこのファイルの内容を貼り付けて実行できます。
収録 LESSON: LESSON 11, LESSON 12, LESSON 13, LESSON 14, LESSON 15, LESSON 16
"""

# --- 共通の準備(序章のフォント設定セル相当) ---
# Colab では 1 回だけ実行すれば OK。日本語ラベルの豆腐(□)化を防ぐ。
!pip install -q matplotlib-fontja
import matplotlib_fontja  # noqa: F401
import matplotlib.pyplot as plt


# ===== LESSON 11 サイコロの6は、本当に6回に1回出るのか =====

import numpy as np

rng = np.random.default_rng(42)

for n in [10, 100, 100000]:
    me = rng.integers(1, 7, size=n)          # サイコロを n 回振る
    wariai = np.mean(me == 6)                # 6が出た割合
    print(f"{n:>6}回: 6が出た回数 {np.sum(me == 6):>5}回 → 割合 {wariai:.4f}")

rng = np.random.default_rng(42)
me = rng.integers(1, 7, size=100000)
for kao in [1, 2, 3, 4, 5, 6]:
    print(f"{kao}の目の割合: {np.mean(me == kao):.4f}")

rng = np.random.default_rng(42)
ikasama = rng.choice([1, 2, 3, 4, 5, 6], size=100000,
                     p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
for kao in [1, 6]:
    print(f"{kao}の目の割合: {np.mean(ikasama == kao):.4f}")

# ===== LESSON 12 何通りあるかを数える: 並べると選ぶ =====

from itertools import permutations, combinations

meibo = ["A", "B", "C"]

print("委員長と副委員長(並べる):")
for kumi in permutations(meibo, 2):
    print(kumi)

print("掃除当番2人(選ぶだけ):")
for kumi in combinations(meibo, 2):
    print(kumi)

import math

print(f"3人から2人を並べる: {math.perm(3, 2)}通り")
print(f"3人から2人を選ぶ:   {math.comb(3, 2)}通り")
print(f"40人から掃除当番2人を選ぶ: {math.comb(40, 2)}通り")
print(f"40人全員を1列に並べる: {math.factorial(40):.2e}通り")

for erabu in [1, 2, 3, 5, 10, 20]:
    print(f"40人から{erabu:>2}人を選ぶ: {math.comb(40, erabu):,}通り")

# ===== LESSON 13 くじは先に引くほうが得なのか =====

rng = np.random.default_rng(42)

kuji = np.tile([1, 0, 0], (100000, 1))   # 当たり1・はずれ2の束を10万セット
kuji = rng.permuted(kuji, axis=1)        # 1セットずつシャッフルして配る
print(f"1番目に引いた人の当たり率: {kuji[:, 0].mean():.4f}")
print(f"2番目に引いた人の当たり率: {kuji[:, 1].mean():.4f}")
print(f"3番目に引いた人の当たり率: {kuji[:, 2].mean():.4f}")

rng = np.random.default_rng(42)
atari = rng.random((100000, 5)) < 0.2    # 当たり20%のガチャを5回×10万人
print(f"5回で1回以上当たった人の割合: {atari.any(axis=1).mean():.4f}")
print(f"計算 1 - 0.8 ** 5 = {1 - 0.8 ** 5:.4f}")

rng = np.random.default_rng(42)
kurasu = rng.integers(1, 366, size=(100000, 23))   # 23人クラスを10万個
# np.unique で重複を除く。種類が23未満なら、誰かの誕生日がかぶっている
kaburi = np.mean([len(np.unique(k)) < 23 for k in kurasu])
print(f"誕生日がかぶるクラスの割合: {kaburi:.4f}")

import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
ninzuu = np.arange(2, 61)
wariai = []
for k in ninzuu:
    kurasu = rng.integers(1, 366, size=(2000, k))
    wariai.append(np.mean([len(np.unique(x)) < k for x in kurasu]))

plt.figure(figsize=(7, 4))
plt.plot(ninzuu, wariai, marker="o", markersize=3, color="black")
plt.axhline(0.5, linestyle="--", color="gray")
plt.axvline(23, linestyle=":", color="gray")
plt.xlabel("クラスの人数")
plt.ylabel("誕生日がかぶる割合")
plt.show()

# ===== LESSON 14 情報が増えると、確率は変わる =====

rng = np.random.default_rng(42)

n = 100000                                   # 10万日分の天気を作る
kumori = rng.random(n) < 0.4                 # 朝くもりの日は40%
ame_ritsu = np.where(kumori, 0.45, 0.05)     # くもりなら45%、晴れなら5%で雨
ame = rng.random(n) < ame_ritsu

print(f"雨が降った日の割合: {ame.mean():.3f}")

kumori_dake = ame[kumori]                    # 朝くもりだった日だけに絞る
print(f"朝くもりだった日: {kumori.sum()}日")
print(f"そのうち雨が降った割合: {kumori_dake.mean():.3f}")

ame_dake = kumori[ame]                       # 雨が降った日だけに絞る
print(f"雨の日のうち、朝くもりだった割合: {ame_dake.mean():.3f}")

# ===== LESSON 15 検査で陽性。本当に病気の確率は? =====

rng = np.random.default_rng(42)

n = 100000                                   # 10万人が検査を受ける
byouki = rng.random(n) < 0.01                # 病気の人は1%
yousei_ritsu = np.where(byouki, 0.99, 0.01)  # 病気なら99%、健康でも1%陽性
yousei = rng.random(n) < yousei_ritsu

print(f"病気の人: {byouki.sum()}人")
print(f"陽性の人: {yousei.sum()}人")

yousei_dake = byouki[yousei]                 # 陽性だった人だけに絞る
print(f"陽性のうち本当に病気の人: {yousei_dake.sum()}人")
print(f"陽性のうち本当に病気の割合: {yousei_dake.mean():.3f}")

rng = np.random.default_rng(42)
byouki = rng.random(n) < 0.001               # 有病率を0.1%に下げる
yousei = rng.random(n) < np.where(byouki, 0.99, 0.01)
print(f"陽性のうち本当に病気の割合: {byouki[yousei].mean():.3f}")

# ===== LESSON 16 コインは覚えていない: 独立とモンティ・ホール =====

rng = np.random.default_rng(42)

omote = rng.integers(0, 2, size=1000000)     # コイン100万回(1=表)
go_ren = (omote[:-5] * omote[1:-4] * omote[2:-3]
          * omote[3:-2] * omote[4:-1]) == 1  # 掛け算が1=表5連続の場所
tsugi = omote[5:][go_ren]                    # その直後の1回だけ集める

print(f"表が5連続した回数: {len(tsugi)}回")
print(f"その直後も表だった割合: {tsugi.mean():.4f}")

rng = np.random.default_rng(42)

n = 100000
atari = rng.integers(0, 3, size=n)           # 当たりのドア(0,1,2)
erabu = rng.integers(0, 3, size=n)           # 最初に選ぶドア

print(f"変えない派の勝率: {np.mean(atari == erabu):.4f}")
print(f"変える派の勝率:   {np.mean(atari != erabu):.4f}")

rng = np.random.default_rng(42)
n = 10000
aru = rng.random(n) < 0.05        # 「本当にそう」の割合(自分の場面に変える)
kenchi = rng.random(n) < np.where(aru, 0.9, 0.1)
print(f"検知した件数: {kenchi.sum()}件")
print(f"検知のうち本物の割合: {aru[kenchi].mean():.2f}")

