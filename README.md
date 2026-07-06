---
title: outing-planner
emoji: ☀️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# ☀️ お出かけプランナー

気象データをもとに、その日の天候に適したお出かけカテゴリをAIが予測するWebアプリです。
Python授業用のサンプルとして、機械学習（scikit-learn）とWebアプリ（Gradio）の基本を学べる最小構成になっています。

## アプリ概要

気温・降水確率・風速・湿度の4つを入力すると、学習済みの機械学習モデル（ランダムフォレスト）がおすすめのお出かけカテゴリを予測します。予測結果と一緒に、おすすめ理由（ルールベースで生成）とおすすめスポット例も表示します。

| ラベル | 表示名 | おすすめスポット例 |
| --- | --- | --- |
| outdoor | 屋外観光 | 公園、神社、海辺、動物園 |
| indoor | 屋内観光 | 水族館、美術館、博物館、映画館 |
| relax | リラックス | カフェ、温泉、図書館、スパ |

## 使用技術

- Python 3.10 以上を推奨
- Gradio（Web UI）
- pandas / NumPy（データ処理）
- scikit-learn（機械学習：RandomForestClassifier）
- joblib（モデルの保存・読み込み）

## ファイル構成

```text
outing-planner/
├─ app.py            # Gradioアプリ本体（UIと予測処理）
├─ train_model.py    # モデル学習スクリプト（サンプルデータ作成→学習→保存）
├─ requirements.txt  # 必要なライブラリ一覧
├─ README.md         # このファイル
└─ model/
   └─ outing_model.pkl  # 学習済みモデル（train_model.py 実行で作成される）
```

## セットアップ方法

必要なライブラリをインストールします。

```bash
pip install -r requirements.txt
```

※ 仮想環境（venv）を使う場合は、先に作成・有効化してから実行してください。

## モデル学習方法

サンプルデータの作成からモデルの学習・保存までを、次のコマンド1つで行います。

```bash
python train_model.py
```

実行すると、`model/outing_model.pkl` に学習済みモデルが保存されます。
学習データはスクリプト内で自動生成しているため、外部データセットの準備は不要です。

※ 同梱の `outing_model.pkl` を使う場合でも、scikit-learn のバージョン違いによる警告を避けるため、最初に一度 `train_model.py` を実行し直すのがおすすめです。

## アプリ起動方法

モデルを作成したあと、アプリを起動します。

```bash
python app.py
```

起動すると、ターミナルにローカルURL（例：`http://127.0.0.1:7860`）が表示されるので、ブラウザで開いてください。

### まとめ（最初から動かす流れ）

```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

## 今後の拡張について（メモ）

コードは関数ごとに分けてあるため、あとから外部APIと連携しやすい構成です。

- `predict_category()` … 天気APIから取得した実際の気象データを渡すように差し替え可能
- `CATEGORY_SPOTS` … Google Maps API などで動的にスポットを取得する処理に置き換え可能
- `build_reason()` … LLMによる理由生成に置き換え可能

※ 今回はAPIサーバー・天気API・Google Maps APIは使用しません。
