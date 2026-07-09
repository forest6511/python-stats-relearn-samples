"""第8章 相関と回帰
『Pythonで学び直す統計・確率』サンプルコード。
Google Colab にこのファイルの内容を貼り付けて実行できます。
収録 LESSON: LESSON 41, LESSON 42, LESSON 43, LESSON 44, LESSON 45
"""

# --- 共通の準備(序章のフォント設定セル相当) ---
# Colab では 1 回だけ実行すれば OK。日本語ラベルの豆腐(□)化を防ぐ。
!pip install -q matplotlib-fontja
import matplotlib_fontja  # noqa: F401
import matplotlib.pyplot as plt


# ===== LESSON 41 一緒に動くかを測る　散布図と相関係数 =====

import numpy as np
import matplotlib.pyplot as plt

temp = np.array([22, 25, 28, 30, 31, 33, 35, 27, 24, 29, 32, 34])  # 気温
ice = np.array([120, 145, 170, 210, 230, 260, 300, 165, 140,
                195, 250, 280])                                    # 売上(千円)

plt.figure(figsize=(6, 4))
plt.scatter(temp, ice, color="black")   # 点を打つだけ
plt.xlabel("気温(度)")
plt.ylabel("アイス売上(千円)")
plt.show()

r = np.corrcoef(temp, ice)[0, 1]        # 相関係数を取り出す
print(f"相関係数 r = {r:.3f}")

rng = np.random.default_rng(42)
noise = rng.normal(0, 60, size=12)   # 大きなブレを作る
ice_bareta = ice + noise             # 売上をばらけさせる
r2 = np.corrcoef(temp, ice_bareta)[0, 1]
print(f"ばらけさせた後の相関係数 r = {r2:.3f}")

# ===== LESSON 42 相関係数だけ見てはいけない　アンスコムの四つ組 =====

import numpy as np

x = np.arange(1, 11)        # 1から10
y = np.arange(1, 11)        # yもx と同じ(完全な直線)
r_before = np.corrcoef(x, y)[0, 1]

y_hazure = y.copy().astype(float)
y_hazure[-1] = -30          # 最後の1点だけ大きく外す
r_after = np.corrcoef(x, y_hazure)[0, 1]

print(f"外れ値なし: r = {r_before:.3f}")
print(f"外れ値1点あり: r = {r_after:.3f}")

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.scatter(x, y, color="black")
ax1.set_title("外れ値なし  r=1.00")
ax2.scatter(x[:-1], y_hazure[:-1], color="black")
ax2.scatter(x[-1], y_hazure[-1], facecolors="none",
            edgecolors="black", s=120)   # 外れ値を丸で強調
ax2.set_title("1点足すと r=-0.32")
plt.show()

r_torinozoku = np.corrcoef(x[:-1], y_hazure[:-1])[0, 1]
print(f"外れ値を取り除くと: r = {r_torinozoku:.3f}")

# ===== LESSON 43 相関は原因ではない　交絡という落とし穴 =====

import numpy as np

rng = np.random.default_rng(42)
n = 200
kion = rng.normal(25, 5, n)                    # 裏で動く気温
ice = 5 * kion + rng.normal(0, 10, n)          # 売上は気温だけから作る
obore = 0.3 * kion + rng.normal(0, 2, n)       # 水難も気温だけから作る

r = np.corrcoef(ice, obore)[0, 1]
print(f"アイスと水難事故の相関: r = {r:.3f}")

chikai = (kion > 23) & (kion < 27)             # 気温が近い日だけ選ぶ
r_soro = np.corrcoef(ice[chikai], obore[chikai])[0, 1]
print(f"気温23〜27度に絞ると: r = {r_soro:.3f}  (該当 {chikai.sum()} 日)")

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.scatter(ice, obore, color="0.6", s=15)
ax1.set_title("全体  r=0.48")
ax2.scatter(ice[chikai], obore[chikai], color="black", s=20)
ax2.set_title("気温をそろえると  r=0.03")
for ax in (ax1, ax2):
    ax.set_xlabel("アイス売上")
    ax.set_ylabel("水難事故")
plt.show()

# ===== LESSON 44 関係から予測する　単回帰と決定係数 =====

import numpy as np

temp = np.array([22, 25, 28, 30, 31, 33, 35, 27, 24, 29, 32, 34])
ice = np.array([120, 145, 170, 210, 230, 260, 300, 165, 140,
                195, 250, 280])

katamuki, seppen = np.polyfit(temp, ice, 1)   # 直線の傾きと切片
print(f"傾き = {katamuki:.2f}   切片 = {seppen:.2f}")

yosoku = katamuki * 30 + seppen               # 気温30度のときの予測
print(f"気温30度のときの売上予測 = {yosoku:.0f}千円")

yhat = katamuki * temp + seppen               # 直線による予測値
zure2 = ((ice - yhat) ** 2).sum()             # 直線からのズレ2乗の合計
zenzure2 = ((ice - ice.mean()) ** 2).sum()    # 平均からのズレ2乗の合計
r2 = 1 - zure2 / zenzure2
print(f"決定係数 R^2 = {r2:.3f}")

yosoku45 = katamuki * 45 + seppen
print(f"気温45度の売上予測 = {yosoku45:.0f}千円")

# ===== LESSON 45 直線に引っかからない　平均への回帰 =====

import numpy as np

rng = np.random.default_rng(42)
jitsuryoku = rng.normal(50, 10, 100)          # 100人の実力(2年間不変)
year1 = jitsuryoku + rng.normal(0, 10, 100)   # 1年目 = 実力 + その年の運
year2 = jitsuryoku + rng.normal(0, 10, 100)   # 2年目 = 実力 + 別の年の運

top10 = np.argsort(year1)[-10:]               # 1年目の上位10人を選ぶ
print(f"上位10人の1年目平均: {year1[top10].mean():.1f}")
print(f"上位10人の2年目平均: {year2[top10].mean():.1f}")
sagatta = (year2[top10] < year1[top10]).sum()
print(f"上位10人のうち2年目に下がった人数: {sagatta}人")

worst10 = np.argsort(year1)[:10]              # 1年目の下位10人
print(f"下位10人の1年目平均: {year1[worst10].mean():.1f}")
print(f"下位10人の2年目平均: {year2[worst10].mean():.1f}")

