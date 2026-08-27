# 🌐 Web アプリと REST API ｜ 使い方

`webapp.py`（Flask）は、**画面**と **REST API** を1つのサーバで動かします。
Gradio 版（`app.py`）と同じモデル・同じ文言を使っていて、違うのは見た目だけです。
予測はどちらも [`outing_ml/serve.py`](../outing_ml/serve.py) を通るので、入力の検証も同じように効きます。

---

## 1. 用意する（初回だけ）

### 1-1. ライブラリを入れる

```bash
# 仮想環境を作る（推奨。すでに作ってあるなら飛ばしてよい）
python -m venv .venv

# 有効にする
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows (PowerShell)

pip install -r requirements.txt
```

Python は **3.11 以上**を想定しています（`python --version` で確認できます）。

### 1-2. データとモデルを作る

Web アプリはモデルを読み込んでから起動します。無いまま起動すると、
「先に python train_all.py を実行してください」と出て止まります。

```bash
python fetch_weather.py    # 気象データをダウンロード（初回のみ・数十秒）
python train_all.py        # 4つのモデルをまとめて学習（20秒ほど）
```

うまくいくと `model/` に4つの `.pkl` と `registry.json` ができます。

---

## 2. 起動する

```bash
python webapp.py
```

こう表示されたら起動できています。

```text
Web アプリを起動します: http://127.0.0.1:5000/
REST API の確認:        http://127.0.0.1:5000/api/health
 * Serving Flask app 'webapp'
```

ブラウザで <http://127.0.0.1:5000/> を開いてください。

### 起動できているかを確かめる

別のターミナルで叩きます。`"ok": true` が返れば動いています。

```bash
curl http://127.0.0.1:5000/api/health
```

```json
{"ok": true, "categories": ["indoor","outdoor","relax"],
 "models": {"category": "v1", "comfort": "v1", "weather_type": "v1"},
 "forecast": "v1", "data_sha256": "84acade580d3714b…"}
```

### 起動のしかたを変える

| やりたいこと | コマンド |
| --- | --- |
| ポートを変える | `PORT=8000 python webapp.py` |
| 同じ端末の別のブラウザから見る | そのままで OK（`127.0.0.1`） |
| 同じネットワークの別の機械から見る | `flask --app webapp:create_web_app run --host 0.0.0.0 --port 5000` |
| コードを直したら自動で再読み込み | `flask --app webapp:create_web_app run --debug` |

> ⚠️ `python webapp.py` で立ち上がるのは**開発用サーバ**です。同時に何人も使う用途には向きません。
> 人に配るときは下の「本番っぽく動かす」を見てください。

### 本番っぽく動かす

```bash
pip install gunicorn
gunicorn "webapp:create_web_app()" --bind 0.0.0.0:8000 --workers 2 --timeout 120
```

- `--workers 2` … モデルは1つあたり数MBなので、ワーカーの数だけメモリを使います
- `--timeout 120` … プラン作成は OpenStreetMap を待つため10秒ほどかかることがあります

### 止める

起動したターミナルで **Ctrl+C**。ポートが埋まったままになったときは次で確認できます。

```bash
lsof -i :5000        # 使っているプロセスを調べる（macOS / Linux）
```

---

## 3. 画面

| URL | できること |
| --- | --- |
| `/` | 天気4項目を入れて予測する。都市を選んで「あしたの予報から」も選べる |
| `/predict` | 予測結果（カテゴリ・確信度・おでかけ日和度・天気タイプ・確率の内訳） |
| `/forecast?city=東京` | あしたの天気の予測（予測区間つき）と、そのおすすめ |
| `/plan` | 時間つきのお出かけプラン（実在するスポットを探す） |
| `/models` | いま動いているモデルの版・成績・学習に使ったデータの指紋 |

画面の色は、見る人の設定（明るい／暗い）に合わせて切り替わります。

---

## 4. REST API

すべて `/api` の下にあります。返り値は必ず `ok` を持ちます。

### 一覧

| メソッド | パス | 内容 |
| --- | --- | --- |
| GET | `/api/health` | 読み込めているモデルと版 |
| GET | `/api/models` | 学習の履歴（版・成績・データの指紋） |
| GET | `/api/cities` | 翌日予報を出せる都市 |
| GET | `/api/weather-types` | 天気タイプの一覧 |
| POST | `/api/predict` | 天気4項目 → おすすめ |
| POST | `/api/predict/batch` | まとめて予測（最大100件） |
| GET | `/api/forecast?city=東京` | あしたの天気とおすすめ |
| POST | `/api/plan` | 時間つきのお出かけプラン |

### 予測する

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 22, "rain_probability": 10, "wind_speed": 2, "humidity": 50}'
```

```json
{
  "ok": true,
  "input": {"temperature": 22.0, "rain_probability": 10.0, "wind_speed": 2.0, "humidity": 50.0},
  "category": "outdoor",
  "probabilities": {"indoor": 0.1065, "outdoor": 0.8495, "relax": 0.044},
  "confidence": 0.8495,
  "comfort_score": 91.5,
  "weather_type": 3,
  "weather_type_name": "過ごしやすい晴れの日",
  "warnings": [],
  "model_versions": {"category": "v1", "comfort": "v1", "weather_type": "v1"}
}
```

### まとめて予測する

```bash
curl -X POST http://127.0.0.1:5000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"days": [
        {"temperature": 22, "rain_probability": 10, "wind_speed": 2, "humidity": 50},
        {"temperature": 15, "rain_probability": 90, "wind_speed": 4, "humidity": 85}
      ]}'
```

### あしたの予報

`city` は日本語なので、`curl` では **URL エンコードが要ります**（ブラウザは自動でやります）。

```bash
curl "http://127.0.0.1:5000/api/forecast?city=%E6%9D%B1%E4%BA%AC"   # 東京
```

```json
{"ok": true, "city": "東京", "base_date": "2026-08-27", "target_date": "2026-08-28",
 "weather": {"temperature": 29.6, "rain_probability": 30.4, "wind_speed": 3.1, "humidity": 64.3},
 "interval": {"temperature": {"low": 28.1, "high": 31.6}, "…": {}},
 "recommendation": {"category": "indoor", "…": {}}}
```

### プランを作る

```bash
curl -X POST http://127.0.0.1:5000/api/plan \
  -H "Content-Type: application/json" \
  -d '{"category": "outdoor", "area": "京都市", "start_time": "10:00", "end_time": "15:00", "radius_km": 3}'
```

スポットは Google Maps →（使えなければ）OpenStreetMap の順に探します。
OpenStreetMap は共有サーバなので**5〜10秒かかります**。

### エラーの返り方

`ok` が `false` のとき、`error.message` に理由が入ります。
直せるものは `error.details` に手がかりも入れています。

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" -d '{"temperature": 22}'
```

```json
{"ok": false,
 "error": {"message": "天気の項目が足りません",
           "details": {"missing": ["rain_probability", "wind_speed", "humidity"],
                       "required": ["temperature", "rain_probability", "wind_speed", "humidity"],
                       "ranges": {"temperature": {"min": -10.0, "max": 40.0}, "…": {}}}}}
```

| 状態 | 意味 |
| --- | --- |
| 400 | 入力が足りない・数値でない・範囲外の指定 |
| 404 | そのURLが無い／地名が見つからない |
| 405 | HTTP メソッドが違う（`/api/predict` は POST） |
| 500 | サーバ側の想定外（中身は返さずログにだけ残す） |
| 503 | 必要なモデルが読み込めていない |

### 範囲を外れた入力

止めずに、丸めたうえで `warnings` を付けて返します。予測を返さないより、
使う側が判断できるほうがよいためです。

```json
{"ok": true, "category": "relax",
 "warnings": ["temperature が入力できる範囲（-10.0〜40.0）の外だったため、40.0 として扱いました",
              "temperature=40.0 は学習データの範囲（-13.6〜36.6）の外です。予測の根拠は弱くなります"]}
```

---

## 5. 予測の記録

予測するたびに `reports/predictions.jsonl` へ1行ずつ残します。

```json
{"at": "2026-08-27T14:12:03+00:00", "kind": "predict",
 "input": {"temperature": 22.0, "…": 0}, "category": "outdoor", "confidence": 0.8495}
```

いまの正解ラベルはルールから作った疑似データです（[doc/README.md](README.md) 第4章）。
実際に使われた天気を貯めておけば、いずれ本物の利用データでモデルを作り直せます。
**残すのは天気の数値と予測結果だけ**で、個人を特定できる情報は入れていません。

止めたいときは環境変数で切れます。

```bash
OUTING_LOG_PREDICTIONS=0 python webapp.py
```

---

## 6. つまずいたら

| 症状 | 原因と直し方 |
| --- | --- |
| `モデルが見つかりません` で起動しない | `python train_all.py` を先に実行する |
| `Address already in use` | 前のサーバが残っている。`lsof -i :5000` で調べて止めるか `PORT=5001 python webapp.py` |
| `/api/forecast` が 400 になる | `curl` で日本語をそのまま URL に入れている。`%E6%9D%B1%E4%BA%AC` のようにエンコードする |
| プラン作成が遅い | OpenStreetMap の利用ルール（1秒に1回）を守って待っているため。2回目以降はキャッシュが効いて速い |
| 地名が見つからない | Google Maps のキーが無くても動くが、`京都市` のように市区町村名で入れるほうが当たりやすい |
| 画面は出るが予測でエラー | `model/` の `.pkl` が古い可能性。`python train_all.py` で作り直す |

---

## 7. Gradio 版（app.py）との違い

| | Gradio（`app.py`） | Flask（`webapp.py`） |
| --- | --- | --- |
| 起動 | `python app.py` | `python webapp.py` |
| ポート | 7860 | 5000 |
| 画面 | 1ページで STEP 1→3 が進む | ページごとに分かれる |
| REST API | 無い | `/api` にある |
| 現在地の取得 | できる（ブラウザの位置情報） | 未対応（地名を入力する） |
| Hugging Face Spaces | ここが動いている | ローカル・自前サーバ向け |

モデルの読み込み・入力の検証・カテゴリの文言は、どちらも同じコードを使っています
（[`outing_ml/serve.py`](../outing_ml/serve.py) と [`presentation.py`](../presentation.py)）。
