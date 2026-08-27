---
title: outing-planner
emoji: ☀️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.11"
app_file: app.py
pinned: false
---

# ☀️ お出かけプランナー

気象データをもとに、その日の天候に適したお出かけカテゴリをAIが予測し、
Google Maps のスポット情報から時間つきのお出かけプランまで作るWebアプリです。
Python授業用のサンプルとして、機械学習（scikit-learn）・Webアプリ（Gradio）・外部API連携の基本を学べる構成になっています。

## アプリ概要

### 1. おすすめカテゴリの予測

気温・降水確率・風速・湿度の4つを入力すると、学習済みの機械学習モデル（勾配ブースティング木）がおすすめのお出かけカテゴリを予測します。予測結果と一緒に、おすすめ理由（ルールベースで生成）とおすすめスポット例も表示します。

モデルは、日本9都市・6年ぶん（19,728日）の実測気象データで学習しています。
どんなモデルなのか・どこまで当たるのか・何が苦手なのかは [doc/README.md](doc/README.md) にまとめてあります。

### 2. お出かけプランの作成

予測のあと、場所と時間を指定すると、その時間内でまわれるタイムスケジュールを作ります。

- **場所の指定**：ブラウザの位置情報から取得した「現在地」か、入力した「市区町村・都道府県」（例：京都市、神奈川県横浜市、沖縄県）のどちらかを選べます
- **時間の指定**：開始時刻と終了時刻（30分きざみ）を選ぶと、その範囲に収まるようにプランを組み立てます
- **行きたい場所の指定**：「はま寿司、水族館」のようにお店・施設の名前を読点区切りで入力すると（4つまで）、その場所を優先してプランの前のほうに入れます
- **スポットの取得**：予測カテゴリに合うキーワード（公園・水族館・カフェ など）で検索し、指定した範囲（1〜20km）の中から実在するスポットを選びます。検索先は **Google Maps →（使えなければ）OpenStreetMap** の順で、どちらも使えないときはGoogleマップの検索リンクを表示します
- お昼の時間帯（11:30〜13:30）にはランチ、2か所まわったあとにはカフェ休憩を自動で入れます。移動時間はスポット間20分で計算します

| ラベル | 表示名 | おすすめスポット例 |
| --- | --- | --- |
| outdoor | 屋外観光 | 公園、神社、海辺、動物園 |
| indoor | 屋内観光 | 水族館、美術館、博物館、映画館 |
| relax | リラックス | カフェ、温泉、図書館、スパ |

## スポット情報の取得先

行き先は次の順に探します。前のものが使えないときだけ、次のものを使います。

| 優先 | 取得先 | 必要なもの | 得られる情報 |
| --- | --- | --- | --- |
| 1 | Google Maps（Places API (New)） | APIキー＋請求先アカウント | 施設名・住所・★評価 |
| 2 | OpenStreetMap（Nominatim） | なし（無料・キー不要） | 施設名・距離 |
| 3 | Googleマップの検索リンク | なし | 検索リンクのみ |

OpenStreetMap は無料の共有サーバーのため、利用ルール（1秒に1回まで・User-Agent を名乗る）を守るよう
`osm_api.py` の中でリクエスト間隔をあけ、同じ検索結果はキャッシュしています。
そのぶん、1回のプラン作成に10秒ほどかかることがあります（2回目以降は一瞬です）。
取得したデータの出典は © OpenStreetMap contributors（ODbL）です。

## Google Maps API の設定（任意）

APIキーを設定しなくてもアプリは動きます（上の表の2番目・3番目が使われます）。
Google Maps の評価つきの情報を使いたいときは、次の手順でAPIキーを設定してください。

1. [Google Cloud Console](https://console.cloud.google.com/) でプロジェクトを作り、**Places API (New)** と **Geocoding API** を有効にする
2. APIキーを発行する（利用するAPIを上の2つに制限しておくと安全です）
3. **プロジェクトに請求先アカウント（クレジットカード）をリンクする**
   （これが無いと、キーが正しくてもすべてのリクエストが拒否されます）
4. キーをアプリに渡す。次のどちらかでOKです

```bash
# A) このフォルダに google_maps_api_key.txt を作って、キーを1行書く（.gitignore済み）

# B) 環境変数に入れる
# Windows (PowerShell)
$env:GOOGLE_MAPS_API_KEY = "取得したキー"
# macOS / Linux
export GOOGLE_MAPS_API_KEY="取得したキー"
```

Hugging Face Spaces で動かす場合は、Space の Settings → Variables and secrets に
`GOOGLE_MAPS_API_KEY` という名前でSecretとして登録します。

※ APIの利用には課金が発生する場合があります。無料枠と利用状況を必ず確認してください。
※ 「現在地から」を使うには、ブラウザで位置情報の利用を許可する必要があります。
　 また、ブラウザの仕様上 `https://` か `http://localhost` でないと現在地を取得できません。

## 使用技術

- Python 3.10 以上を推奨
- Gradio（Web UI）
- pandas / NumPy（データ処理）
- scikit-learn（機械学習：HistGradientBoostingClassifier）
- joblib（モデルの保存・読み込み）

## ファイル構成

```text
outing-planner/
├─ app.py            # Gradioアプリ本体（UIと予測処理）
├─ maps_api.py       # Google Maps API との連携（地名→緯度経度、スポット検索）
├─ osm_api.py        # OpenStreetMap との連携（キー不要のスポット検索）
├─ planner.py        # お出かけプランの組み立て（時間割づくり、スポット割り当て）
├─ fetch_weather.py  # 気象データの取得（Open-Meteo → data/weather_jp.csv）
├─ train_model.py    # モデル学習スクリプト（ラベル付け→モデル選定→学習→保存）
├─ requirements.txt  # 必要なライブラリ一覧
├─ README.md         # このファイル
├─ data/
│  └─ weather_jp.csv    # 学習データ（9都市・6年ぶんの実測気象データ）
├─ doc/
│  ├─ README.md         # モデルカード（モデルの説明書）
│  └─ dataset.md        # データセットの説明書
└─ model/
   ├─ outing_model.pkl  # 学習済みモデル（train_model.py 実行で作成される）
   └─ model_card.json   # 成績や設定の記録（同上）
```

## セットアップ方法

必要なライブラリをインストールします。

```bash
pip install -r requirements.txt
```

※ 仮想環境（venv）を使う場合は、先に作成・有効化してから実行してください。

## モデル学習方法

学習データの用意からモデルの学習・保存までを、次の2つのコマンドで行います。

```bash
python fetch_weather.py    # 気象データをダウンロード（初回のみ・数十秒）
python train_model.py      # ラベル付け → モデル選定 → 学習 → 保存
```

`fetch_weather.py` は [Open-Meteo](https://open-meteo.com/) の過去データAPIから、
日本9都市・6年ぶん（2019〜2024年）の実測気象データを取得して `data/weather_jp.csv` に保存します。
APIキーは不要です（出典：Open-Meteo / ECMWF ERA5、CC BY 4.0）。

`train_model.py` は、5つの候補（ベースライン・ロジスティック回帰・決定木・
ランダムフォレスト・勾配ブースティング）を5分割交差検証で比べて、
いちばん成績の良いモデルを選んで学習します。実行すると次の2つが保存されます。

- `model/outing_model.pkl` … 学習済みモデル（アプリが読み込む）
- `model/model_card.json` … 成績・設定・データの記録

データが無いときは自動でダウンロードするため、`python train_model.py` だけでも動きます。
成績や限界などのくわしい説明は [doc/README.md](doc/README.md)、
学習データそのものの説明は [doc/dataset.md](doc/dataset.md) にあります。

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
python fetch_weather.py
python train_model.py
python app.py
```

## 今後の拡張について（メモ）

コードは関数ごとに分けてあるため、あとから機能を差し替えやすい構成です。

- `predict_category()` … 天気APIから取得した実際の気象データを渡すように差し替え可能
- `model.predict_proba()` … 予測の確率を取り出せるので、「おすすめ度○%」の表示にも使える
- `build_reason()` … LLMによる理由生成に置き換え可能
- `planner.STAY_MINUTES` / `TRAVEL_MINUTES` … 滞在時間・移動時間の目安を調整可能
- `planner.KIND_KEYWORDS` … スポット検索のキーワードを増やすと、行き先のバリエーションが広がる
- `maps_api.search_places()` … Directions API を足せば、実際の移動時間で組み立てることも可能

※ アプリの天気は手入力です（天気予報APIは使っていません）。
　 Open-Meteo を使うのは、モデルの学習データを取得するとき（`fetch_weather.py`）だけです。
