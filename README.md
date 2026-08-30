---
title: outing-planner
emoji: ☀️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 5000
pinned: false
---

# ☀️ お出かけプランナー

気象データをもとに、その日の天候に適したお出かけカテゴリをAIが予測し、
Google Maps のスポット情報から時間つきのお出かけプランまで作るWebアプリです。
Python授業用のサンプルとして、機械学習（scikit-learn）・Webアプリ（Flask）・外部API連携の基本を学べる構成になっています。

## アプリ概要

### 1. おすすめカテゴリの予測

気温・降水確率・風速・湿度の4つを入力すると、学習済みの機械学習モデル（勾配ブースティング木）がおすすめのお出かけカテゴリを予測します。予測結果と一緒に、おすすめ理由（ルールベースで生成）とおすすめスポット例も表示します。

モデルは、全都道府県の県庁所在地47都市・6年ぶん（103,024日）の実測気象データで学習しています。
どんなモデルなのか・どこまで当たるのか・何が苦手なのかは [doc/README.md](doc/README.md) にまとめてあります。

### 3. 4つのモデル

このプロジェクトには、役割のちがう4つのモデルが入っています。
機械学習の3つの型（分類・回帰・教師なし学習）をひと通りためせる構成です。

| # | モデル | 学習の種類 | やること | 成績 | 説明 |
| --- | --- | --- | --- | --- | --- |
| ① | カテゴリ予測 | 教師あり・**分類** | 天気 → outdoor / indoor / relax | 正解率 0.821（上限 0.820） | [doc/README.md](doc/README.md) |
| ② | 翌日の天気予測 | 教師あり・**回帰（時系列）** | きょうまでの天気 → あしたの天気 | 気温 MAE 1.72℃ | [doc/forecast.md](doc/forecast.md) |
| ③ | おでかけ日和度 | 教師あり・**回帰** | 天気 → 0〜100点 | MAE 4.67点（下限 4.62点） | [doc/comfort.md](doc/comfort.md) |
| ④ | 天気タイプ分け | **教師なし**・クラスタリング | 天気 → 3つのタイプ | シルエット係数 0.277 | [doc/weather-types.md](doc/weather-types.md) |

アプリが実際に使っているのは①です。②〜④は、①と同じデータから作った発展用のモデルで、Webアプリの「あしたの予報」「週間予報」「都市を比べる」画面から使えます。

### 4. 画面と REST API

画面と REST API を1つのサーバ（Flask）でまとめて提供します。使い方は [doc/webapp.md](doc/webapp.md) にまとめています。

```bash
python webapp.py    # → http://127.0.0.1:5000
```

| 起動方法 | 特徴 |
| --- | --- |
| `python webapp.py` | 画面（`/`）と REST API（`/api`）が同じサーバで動く |
| `docker compose up --build` | 依存のインストール不要でコンテナのまま起動 |

REST API では、予測・まとめて予測・あしたの予報・週間予報・都市の比較・プラン作成などを JSON で扱えます。

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 22, "rain_probability": 10, "wind_speed": 2, "humidity": 50}'
```

API仕様（OpenAPI 3.0）は `/api/openapi.json`、ブラウザで試すなら `/docs`（Swagger UI）です。

### 2. お出かけプランの作成

予測のあと、場所と時間を指定すると、その時間内でまわれるタイムスケジュールを作ります。

- **場所の指定**：ブラウザの位置情報から取得した「現在地」か、入力した「市区町村・都道府県」（例：京都市、神奈川県横浜市、沖縄県）のどちらかを選べます
- **時間の指定**：開始時刻と終了時刻（30分きざみ）を選ぶと、その範囲に収まるようにプランを組み立てます
- **行きたい場所の指定**：「はま寿司、水族館」のようにお店・施設の名前を読点区切りで入力すると（4つまで）、その場所を優先してプランの前のほうに入れます
- **スポットの取得**：予測カテゴリに合うキーワード（公園・水族館・カフェ など）で検索し、指定した範囲（1〜20km）の中から実在するスポットを選びます。検索先は **Google Maps →（使えなければ）OpenStreetMap** の順で、どちらも使えないときはGoogleマップの検索リンクを表示します
- お昼の時間帯（11:30〜13:30）にはランチ、2か所まわったあとにはカフェ休憩を自動で入れます。移動時間はスポット間20分で計算します
- 作ったプランには共有リンクが発行され、リンクを知っている人なら誰でも同じプランを開けます

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

- Python 3.11 以上を推奨
- Flask（Web UI・REST API）
- pandas / NumPy（データ処理）
- scikit-learn（機械学習：分類・回帰・クラスタリング）
- joblib（モデルの保存・読み込み）

## ファイル構成

```text
outing-planner/
├─ webapp.py         # Flaskアプリ本体（画面＋REST API）
├─ api.py            # REST API（JSONで返す部分）
├─ presentation.py   # 画面の文言（表示名・おすすめ理由・入力欄）
├─ geocoding.py      # 地名・現在地→緯度経度（Google Maps → OpenStreetMap の順）
├─ maps_api.py       # Google Maps API との連携（地名→緯度経度、スポット検索）
├─ osm_api.py        # OpenStreetMap との連携（キー不要のスポット検索）
├─ planner.py        # お出かけプランの組み立て（時間割づくり、スポット割り当て）
├─ shared_plans.py   # プランの共有リンク
├─ prediction_log.py # 予測の記録
├─ monitoring.py     # ドリフト監視
├─ city_comparison.py # 都市の比較（キャッシュ付き）
├─ rate_limit.py     # APIのレート制限
├─ access_log.py     # 構造化アクセスログ
├─ openapi_spec.py   # REST APIの仕様（OpenAPI 3.0）
├─ fetch_weather.py  # 気象データの取得（Open-Meteo → data/weather_jp.csv）
├─ train_all.py      # ①〜④のモデルをまとめて学習する
├─ train_model.py           # ① カテゴリ予測モデル（分類）
├─ train_forecast.py        # ② 翌日の天気予測モデル（時系列の回帰）
├─ train_comfort.py         # ③ おでかけ日和度モデル（回帰）
├─ train_weather_types.py   # ④ 天気タイプ分けモデル（教師なし）
├─ requirements.txt  # 必要なライブラリ一覧
├─ Dockerfile / docker-compose.yml  # コンテナで動かすための設定
├─ README.md         # このファイル
├─ data/
│  ├─ weather_jp.csv         # 学習データ（47都市・6年ぶんの実測気象データ）
│  └─ weather_jp_holdout.csv # 未来データ（学習後の最終確認用）
├─ templates/        # Flaskアプリの画面（HTML）
├─ static/           # Flaskアプリのスタイル（CSS）
├─ doc/
│  ├─ README.md         # モデル一覧＋①のモデルカード
│  ├─ webapp.md         # Webアプリと REST API の使い方
│  ├─ dataset.md        # データセットの説明書
│  ├─ forecast.md       # ② のモデルカード
│  ├─ comfort.md        # ③ のモデルカード
│  ├─ weather-types.md  # ④ のモデルカード
│  ├─ accessibility.md  # アクセシビリティ対応
│  └─ development-report.md  # 開発の経緯
└─ model/
   ├─ outing_model.pkl       # ① 学習済みモデル（アプリが読み込む）
   ├─ model_card.json        # ① 成績や設定の記録
   ├─ forecast_model.pkl     # ② 学習済みモデル
   ├─ forecast_card.json     # ② 成績や設定の記録
   ├─ comfort_model.pkl      # ③ 学習済みモデル
   ├─ comfort_card.json      # ③ 成績や設定の記録
   ├─ weather_type_model.pkl # ④ 学習済みモデル
   ├─ weather_types.json     # ④ タイプの一覧と成績
   └─ registry.json          # 学習の履歴（版・データの指紋）
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
python fetch_weather.py    # 気象データをダウンロード（初回のみ・47都市ぶんで数分かかることがあります）
python train_all.py        # 4つのモデルをまとめて学習（20秒ほど）
```

1つずつ作ることもできます。アプリを動かすだけなら①だけで足ります。

```bash
python train_model.py           # ① カテゴリ予測（アプリが使うのはこれ）
python train_forecast.py        # ② 翌日の天気予測
python train_comfort.py         # ③ おでかけ日和度
python train_weather_types.py   # ④ 天気タイプ分け
```

※ ②〜④は①のモデルを参照するので、`train_model.py` を先に実行してください
（`train_all.py` は正しい順に実行します）。

`fetch_weather.py` は [Open-Meteo](https://open-meteo.com/) の過去データAPIから、
全都道府県の県庁所在地47都市・6年ぶん（2019〜2024年）の実測気象データを取得して `data/weather_jp.csv` に保存します。
APIキーは不要です（出典：Open-Meteo / ECMWF ERA5、CC BY 4.0）。取得元の利用上限に当たった場合は
`python fetch_weather.py --resume` で残りの都市だけ続きから取れます。

`train_model.py` は、5つの候補（ベースライン・ロジスティック回帰・決定木・
ランダムフォレスト・勾配ブースティング）を5分割交差検証で比べて、
いちばん成績の良いモデルを選んで学習します。実行すると次の2つが保存されます。

- `model/outing_model.pkl` … 学習済みモデル（アプリが読み込む）
- `model/model_card.json` … 成績・設定・データの記録

②〜④も同じように、成績や設定を `model/*_card.json`（④は `weather_types.json`）に書き出します。
データが無いときは自動でダウンロードするため、学習スクリプトだけでも動きます。
乱数の種を固定しているので、**何度実行しても同じ結果**になります。
成績や限界などのくわしい説明は [doc/README.md](doc/README.md)、
学習データそのものの説明は [doc/dataset.md](doc/dataset.md) にあります。

※ 同梱の `.pkl` を使う場合でも、scikit-learn のバージョン違いによる警告を避けるため、最初に一度 `train_all.py` を実行し直すのがおすすめです。

## アプリ起動方法

モデルを作成したあと、アプリを起動します。

```bash
python webapp.py
```

起動すると、ターミナルにローカルURL（例：`http://127.0.0.1:5000`）が表示されるので、ブラウザで開いてください。
ポートを変えたいときは `PORT=8000 python webapp.py` のようにします。
起動できているかは `curl http://127.0.0.1:5000/api/health` で確かめられます。
くわしい手順・API仕様・つまずいたときの対処は [doc/webapp.md](doc/webapp.md) にあります。

Docker で動かす場合は次の1コマンドです（依存のインストールは不要）。

```bash
docker compose up --build
```

### まとめ（最初から動かす流れ）

```bash
pip install -r requirements.txt
python fetch_weather.py
python train_all.py
python webapp.py
```

## 今後の拡張について（メモ）

コードは関数ごとに分けてあるため、あとから機能を差し替えやすい構成です。

- `outing_ml.serve.OutingService.predict()` … 天気APIから取得した実際の気象データを渡すように差し替え可能
- `predict_proba()` … 予測の確率を取り出せるので、「おすすめ度○%」の表示にも使える
- `model/comfort_model.pkl` … 「今日のおでかけ日和度 91点」を画面に足せる
- `model/weather_type_model.pkl` … 同じ屋外観光でも、天気タイプで行き先を変えられる
- `model/forecast_model.pkl` … 「あしたのプラン」を作るときの下ごしらえに使える
- `presentation.build_reason()` … LLMによる理由生成に置き換え可能
- `planner.STAY_MINUTES` / `TRAVEL_MINUTES` … 滞在時間・移動時間の目安を調整可能
- `planner.KIND_KEYWORDS` … スポット検索のキーワードを増やすと、行き先のバリエーションが広がる
- `maps_api.search_places()` … Directions API を足せば、実際の移動時間で組み立てることも可能

※ アプリの天気は手入力です（天気予報APIは使っていません）。
　 Open-Meteo を使うのは、モデルの学習データを取得するとき（`fetch_weather.py`）だけです。
