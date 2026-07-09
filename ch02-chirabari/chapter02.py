"""第2章 分散・標準偏差・偏差値
『Pythonで学び直す統計・確率』サンプルコード。
Google Colab にこのファイルの内容を貼り付けて実行できます。
収録 LESSON: LESSON 06, LESSON 07, LESSON 08, LESSON 09, LESSON 10
"""

# --- 共通の準備(序章のフォント設定セル相当) ---
# Colab では 1 回だけ実行すれば OK。日本語ラベルの豆腐(□)化を防ぐ。
!pip install -q matplotlib-fontja
import matplotlib_fontja  # noqa: F401
import matplotlib.pyplot as plt


# ===== LESSON 06 平均が同じ2クラス、どちらが「荒れて」いる? =====

import numpy as np

a = np.array([55, 58, 60, 62, 65])   # A組(まとまっている)
b = np.array([30, 45, 60, 75, 90])   # B組(荒れている)

# 平均からの距離(偏差)を全員分計算する
print(f"A組の偏差: {a - np.mean(a)}")
print(f"B組の偏差: {b - np.mean(b)}")

print(f"A組の偏差の合計: {np.sum(a - np.mean(a))}")
print(f"B組の偏差の合計: {np.sum(b - np.mean(b))}")

# 偏差を2乗して平均する = 分散
print(f"A組の分散: {np.var(a)}")
print(f"B組の分散: {np.var(b)}")

b2 = np.array([55, 45, 60, 75, 90])   # 30点 → 55点
print(f"改善後のB組の分散: {np.var(b2)}")

# ===== LESSON 07 単位を元に戻す: 標準偏差 =====

b = np.array([30, 45, 60, 75, 90])

# 分散にルートを取ると標準偏差
print(f"B組の分散:     {np.var(b)}")
print(f"分散のルート:  {np.sqrt(np.var(b))}")

# 標準偏差は np.std() で一発でも出せる
print(f"B組の標準偏差: {np.std(b)}")

m, s = np.mean(b), np.std(b)
# 平均±標準偏差の範囲に入っている人を数える
inside = np.sum((b >= m - s) & (b <= m + s))
print(f"平均±標準偏差({m-s:.0f}〜{m+s:.0f}点)の人数: {inside}/{len(b)}人")

a = np.array([55, 58, 60, 62, 65])
print(f"A組の標準偏差: {np.std(a):.1f}点")
print(f"B組の標準偏差: {np.std(b):.1f}点")

# ===== LESSON 08 偏差値の正体: 違う科目を同じ土俵で比べる =====

# 数学: 平均70, 標準偏差10 のテストで 80点を取った
score, mean, sd = 80, 70, 10

# 偏差値 = (自分の点 - 平均) / 標準偏差 * 10 + 50
hensachi = (score - mean) / sd * 10 + 50
print(f"偏差値: {hensachi}")

# 数学: 平均85, 標準偏差10 のテストで 90点
math = (90 - 85) / 10 * 10 + 50
# 英語: 平均55, 標準偏差10 のテストで 75点
eng = (75 - 55) / 10 * 10 + 50
print(f"数学90点の偏差値: {math}")
print(f"英語75点の偏差値: {eng}")

from scipy import stats

# 偏差値70 = 標準偏差2つ分上。それより上にいる割合
print(f"偏差値70より上: {(1 - stats.norm.cdf(2)) * 100:.1f}%")
print(f"偏差値60より上: {(1 - stats.norm.cdf(1)) * 100:.1f}%")

# ===== LESSON 09 単位が違うと散らばりは比べられない: 変動係数 =====

tenA = np.array([470, 500, 530, 460, 540, 500])  # 平均500万円あたり
tenB = np.array([70, 100, 130, 60, 140, 100])     # 平均100万円あたり

for name, d in [("店A", tenA), ("店B", tenB)]:
    m, s = np.mean(d), np.std(d)
    cv = s / m   # 変動係数 = 標準偏差 ÷ 平均
    print(f"{name}: 平均{m:.0f}万円 標準偏差{s:.1f}万円 変動係数{cv*100:.1f}%")

shincho = np.array([160, 165, 170, 175, 180])  # 身長(cm)
taiju = np.array([50, 58, 65, 72, 80])           # 体重(kg)

for name, d, unit in [("身長", shincho, "cm"), ("体重", taiju, "kg")]:
    m, s = np.mean(d), np.std(d)
    print(f"{name}: 標準偏差{s:.2f}{unit} 変動係数{s/m*100:.1f}%")

# ===== LESSON 10 分布を5つの数字で要約する: 箱ひげ図 =====

data = np.array([30, 45, 48, 50, 52, 55, 58, 60, 62, 65, 68, 70, 95])

# 四分位数を計算する(25%, 50%, 75%の位置の値)
q1, q2, q3 = np.percentile(data, [25, 50, 75])
iqr = q3 - q1   # 四分位範囲

print(f"第1四分位: {q1}")
print(f"中央値:    {q2}")
print(f"第3四分位: {q3}")
print(f"四分位範囲(IQR): {iqr}")

# 1.5×IQR ルール: 箱から IQR の1.5倍を超えて離れたら外れ値
low = q1 - 1.5 * iqr
high = q3 + 1.5 * iqr
print(f"外れ値の境界: {low}未満 または {high}超")

outliers = data[(data < low) | (data > high)]
print(f"外れ値: {outliers}")

plt.figure(figsize=(7, 3))
plt.boxplot(data, orientation="horizontal")
plt.xlabel("値")
plt.show()

# 数字を自分のデータに書き換える(例: 睡眠時間)
data = np.array([6.5, 7.0, 5.5, 8.0, 6.0, 7.5, 6.5, 5.0, 9.0, 6.5])

print(f"平均:      {np.mean(data):.2f}")
print(f"標準偏差:  {np.std(data):.2f}")
print(f"変動係数:  {np.std(data) / np.mean(data) * 100:.1f}%")

