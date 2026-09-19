# Pythonで学び直す統計・確率 — サンプルコード

**図解と60の問いで、中1から大学までもう一度ゼロからわかる**

書籍『Pythonで学び直す統計・確率』（森川 陽介 著）の全 LESSON の Python コードです。
すべて **Google Colab** にそのまま貼り付けて動きます（インストール不要）。

Kindle 版: [Pythonで学び直す統計・確率](https://www.amazon.co.jp/dp/B0H8C6STJV)

## 使い方

1. [Google Colab](https://colab.research.google.com/) を開き、Google アカウントでログインします（無料。実行にはログインが必要です）
2. 読みたい章のフォルダを開き、`chapterNN.py` を開きます。コード右上の **Raw** ボタンを押すとコードだけの画面になるので、全選択（Windows は Ctrl+A、Mac は Cmd+A）してコピーします
3. Colab の新しいセルに貼り付けて ▶ を押します（▶ が見つからないときは Shift+Enter でも実行できます）

**まずはファイルの中身を全部貼り付けて、上から順に実行してください。そのまま本と同じ出力とグラフが出ます。**

> コードの先頭に `!pip install -q matplotlib-fontja` という行があります。これはグラフの日本語が文字化けしないための準備で、Colab が自動でやってくれます。消さずにそのまま実行してください（数秒で終わります）。

> LESSON ごとに区切って試したいときは、ファイル内の `# ===== LESSON NN ... =====` の区切りを目印に、必要な範囲だけ貼り付けてください。

## 動作環境

- **Google Colab**（推奨・ブラウザだけで動く・自分で環境を作る必要なし）
- ローカルの Python でも動きます（Python 3.10 以上）。その場合は `numpy` `scipy` `pandas` `matplotlib` `statsmodels` を入れてください（`matplotlib-fontja` はファイル冒頭の行が自動で入れます）

統計の計算に使うライブラリはどれも Colab に最初から入っているので、あなたが手でインストールする作業はありません。

## 章の一覧

- [`ch01-daihyouchi/`](./ch01-daihyouchi/) — 第1章 平均・中央値・最頻値（5 LESSON）
- [`ch02-chirabari/`](./ch02-chirabari/) — 第2章 分散・標準偏差・偏差値（5 LESSON）
- [`ch03-kakuritsu/`](./ch03-kakuritsu/) — 第3章 確率（6 LESSON）
- [`ch04-bunpu/`](./ch04-bunpu/) — 第4章 分布（6 LESSON）
- [`ch05-hyouhon-suitei/`](./ch05-hyouhon-suitei/) — 第5章 標本・大数の法則・中心極限定理（7 LESSON）
- [`ch06-kentei-ttest/`](./ch06-kentei-ttest/) — 第6章 仮説検定とt検定（7 LESSON）
- [`ch07-chi-square/`](./ch07-chi-square/) — 第7章 カイ二乗検定（4 LESSON）
- [`ch08-soukan-kaiki/`](./ch08-soukan-kaiki/) — 第8章 相関と回帰（5 LESSON）
- [`ch09-anova/`](./ch09-anova/) — 第9章 分散分析(ANOVA)（4 LESSON）
- [`ch10-jukaiki/`](./ch10-jukaiki/) — 第10章 重回帰分析（4 LESSON）
- [`ch11-logistic/`](./ch11-logistic/) — 第11章 ロジスティック回帰（3 LESSON）
- [`ch12-bayes/`](./ch12-bayes/) — 第12章 ベイズ統計入門（3 LESSON）
- [`ch13-sougou-enshuu/`](./ch13-sougou-enshuu/) — 第13章 総合演習（1 LESSON）

## この本について

数式アレルギーで統計の本を挫折した人が、**中学レベルの問いから始めて、Python で手を動かしながら t 検定・回帰・分散分析・ベイズまで**たどり着くための本です。
各 LESSON は「日常の問い → 図解 → Python で確かめる → 数式で整理」の順で進みます。

## ライセンス

このリポジトリのコードは、書籍の読者が自由に写経・改変・実行して構いません（MIT License）。

