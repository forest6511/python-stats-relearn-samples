"""第1章 平均・中央値・最頻値
『Pythonで学び直す統計・確率』サンプルコード。
Google Colab にこのファイルの内容を貼り付けて実行できます。
収録 LESSON: LESSON 01, LESSON 02, LESSON 03, LESSON 04, LESSON 05
"""

# --- 共通の準備(序章のフォント設定セル相当) ---
# Colab では 1 回だけ実行すれば OK。日本語ラベルの豆腐(□)化を防ぐ。
!pip install -q matplotlib-fontja
import matplotlib_fontja  # noqa: F401
import matplotlib.pyplot as plt


# ===== LESSON 01 平均所得536万円なのに、「そんなにもらってない」のはなぜ? =====

import numpy as np

# 社員9人と社長の年収(万円)。最後の1人が社長
zenin = [250, 280, 300, 320, 350, 380, 400, 420, 450, 3000]

print(f"社長を除く9人の平均: {np.mean(zenin[:9])} 万円")
print(f"社長を入れた10人の平均: {np.mean(zenin)} 万円")

zenin2 = [250, 280, 300, 320, 350, 380, 400, 420, 450, 10000]
print(f"社長が1億円のときの平均: {np.mean(zenin2)} 万円")

# ===== LESSON 02 「普通の人」の値が知りたい: 中央値 =====

zenin = [250, 280, 300, 320, 350, 380, 400, 420, 450, 3000]

print(f"並べ替え: {sorted(zenin)}")
print(f"中央値: {np.median(zenin)} 万円")
print(f"平均:   {np.mean(zenin)} 万円")

zenin2 = [250, 280, 300, 320, 350, 380, 400, 420, 450, 10000]
print(f"社長が1億円のときの中央値: {np.median(zenin2)} 万円")

# ===== LESSON 03 靴屋の仕入れはなぜ平均サイズではダメか: 最頻値 =====

import statistics

kutsu = [24.0, 24.5, 24.5, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0,
         25.5, 25.5, 25.5, 26.0, 26.5, 27.0]

print(f"最頻値: {statistics.mode(kutsu)} cm")
print(f"平均:   {round(np.mean(kutsu), 2)} cm")
print(f"中央値: {np.median(kutsu)} cm")

kutsu2 = [24.0, 24.5, 24.5, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0,
          25.5, 25.5, 25.5, 26.0, 26.5, 28.0]
print(f"最頻値: {statistics.mode(kutsu2)} cm")
print(f"平均:   {round(np.mean(kutsu2), 2)} cm")

# ===== LESSON 04 1万人の数字は「束ねて」見る: 度数分布表とヒストグラム =====

import pandas as pd

rng = np.random.default_rng(42)
# 年収っぽい形の架空データを1万人分作る(詳細は第4章)
nenshu = rng.lognormal(np.log(400), 0.6, size=10000)

print(f"平均:   {round(np.mean(nenshu), 1)} 万円")
print(f"中央値: {round(np.median(nenshu), 1)} 万円")

# 階級(バケツ)を決めて、各階級の人数を数える
kaikyuu = pd.cut(nenshu, bins=[0, 200, 400, 600, 800, 1000, 20000])
print(kaikyuu.value_counts().sort_index())

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(nenshu, bins=5, edgecolor="white")
axes[0].set_title("bins=5")
axes[0].set_xlabel("年収(万円)")
axes[0].set_ylabel("人数")
axes[1].hist(nenshu, bins=50, edgecolor="white")
axes[1].set_title("bins=50")
axes[1].set_xlabel("年収(万円)")
axes[1].set_ylabel("人数")
plt.tight_layout()
plt.show()

# ===== LESSON 05 本物の日本の所得分布を描く =====

# 2024(令和6)年 国民生活基礎調査: 所得金額階級別の世帯の割合(%)
# 左から「100万円未満」「100〜200万円」…「1900〜2000万円」「2000万円以上」
wariai = [6.7, 14.4, 14.4, 13.1, 9.9, 8.5, 7.6, 5.4, 4.4, 3.3,
          2.8, 2.0, 1.6, 1.0, 0.9, 0.8, 0.6, 0.4, 0.4, 0.4, 1.4]
kaikyuu = np.arange(50, 2101, 100)   # 各階級の真ん中の値

plt.figure(figsize=(9, 4.5))
plt.bar(kaikyuu, wariai, width=95, edgecolor="white")
plt.axvline(536, color="black", linestyle="--", linewidth=2,
            label="平均 536万円")
plt.axvline(410, color="black", linestyle=":", linewidth=2,
            label="中央値 410万円")
plt.xlabel("所得金額(万円)")
plt.ylabel("世帯の割合(%)")
plt.legend()
plt.show()

# 金額を自分のデータに書き換える
shishutsu = [480, 1200, 350, 5800, 700, 980, 350, 2400, 620, 350]
print(f"平均:   {np.mean(shishutsu)} 円")
print(f"中央値: {np.median(shishutsu)} 円")
print(f"最頻値: {statistics.mode(shishutsu)} 円")

