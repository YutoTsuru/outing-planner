"""
お出かけプランナー：Gradioアプリ

気象データ（気温・降水確率・風速・湿度）を入力すると、
学習済みモデルがおすすめのお出かけカテゴリを予測して表示します。
そのあと、現在地または指定した市区町村・都道府県のまわりから
Google Maps でスポットを探して、時間つきのお出かけプランを作ります。

画面は「STEP 1 → STEP 2 → STEP 3」の順に進み、
前のステップが終わると次のカードが表示されます。

起動前に train_model.py を実行して、モデルを作成しておいてください。
Google Maps 連携を使うときは、このフォルダに google_maps_api_key.txt を作って
APIキーを書くか、環境変数 GOOGLE_MAPS_API_KEY に設定します。
（未設定でもアプリは動きます。その場合はGoogleマップの検索リンクを表示します）

実行方法:
    python app.py
"""

import os

import gradio as gr
import joblib
import pandas as pd

import maps_api
import osm_api
import planner

# ---------------------------------------------------------------
# 設定・固定データ
# ---------------------------------------------------------------

# モデルの場所（train_model.py と合わせる）
MODEL_PATH = os.path.join("model", "outing_model.pkl")

# 特徴量の列名（train_model.py と同じ順番にすること）
FEATURE_COLUMNS = ["temperature", "rain_probability", "wind_speed", "humidity"]

# カテゴリの (アイコン, 表示名)
CATEGORY_LABELS = {
    "outdoor": ("🌳", "屋外観光"),
    "indoor": ("🏛️", "屋内観光"),
    "relax": ("☕", "リラックス"),
}

# カテゴリごとのおすすめスポット例（Google Maps を使わないときの表示用）
CATEGORY_SPOTS = {
    "outdoor": ["公園", "神社", "海辺", "動物園"],
    "indoor": ["水族館", "美術館", "博物館", "映画館"],
    "relax": ["カフェ", "温泉", "図書館", "スパ"],
}

# 場所の決め方（ラジオボタンの選択肢）
LOCATION_MODE_GPS = "現在地から"
LOCATION_MODE_AREA = "市区町村・都道府県から"

# ブラウザの位置情報を取得する JavaScript
# （Gradio では js= に書いた関数の戻り値が、そのまま outputs に入る）
# ※ 位置情報は https か localhost でしか取得できない点に注意
GET_LOCATION_JS = """
async () => {
    if (!navigator.geolocation) {
        return ["", "", "この端末では現在地を取得できません。"];
    }
    try {
        const position = await new Promise((resolve, reject) => {
            navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 10000 });
        });
        const latitude = position.coords.latitude;
        const longitude = position.coords.longitude;
        return [
            String(latitude),
            String(longitude),
            "現在地を取得しました（" + latitude.toFixed(4) + ", " + longitude.toFixed(4) + "）",
        ];
    } catch (error) {
        return ["", "", "取得できませんでした。ブラウザで位置情報の利用を許可してください。"];
    }
}
"""


def scroll_to_js(element_id):
    """指定したIDの場所まで、なめらかにスクロールする JavaScript を作る。"""
    return """
() => {
    const target = document.querySelector("#ELEMENT_ID");
    if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}
""".replace("ELEMENT_ID", element_id)


# ---------------------------------------------------------------
# モデル関連の関数
# ---------------------------------------------------------------

def load_model():
    """学習済みモデルを読み込む。"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"モデルが見つかりません: {MODEL_PATH}\n"
            "先に python train_model.py を実行してください。"
        )
    return joblib.load(MODEL_PATH)


# アプリ起動時に1回だけモデルを読み込む
model = load_model()


def predict_category(temperature, rain_probability, wind_speed, humidity):
    """入力値からカテゴリ（outdoor / indoor / relax）を予測する。"""
    input_df = pd.DataFrame(
        [[temperature, rain_probability, wind_speed, humidity]],
        columns=FEATURE_COLUMNS,
    )
    return model.predict(input_df)[0]


def build_reason(label, temperature, rain_probability, wind_speed, humidity):
    """予測結果と入力値から、おすすめ理由の文章を作る（ルールベース）。"""
    if label == "outdoor":
        return "降水確率が低く、気温も過ごしやすいため、屋外観光に向いています。"

    if label == "indoor":
        if rain_probability >= 50:
            return "雨の可能性が高いため、屋内で楽しめる場所がおすすめです。"
        if wind_speed >= 10:
            return "風が強いため、屋内で快適に過ごせる場所がおすすめです。"
        return "外で過ごしにくい天気のため、屋内のスポットがおすすめです。"

    # ここから下は relax の場合
    if humidity >= 80:
        return "湿度が高いので、ゆっくり過ごせる場所がおすすめです。"
    if temperature < 10:
        return "気温が低めなので、あたたかい場所でのんびり過ごすのがおすすめです。"
    return "今日は無理せず、リラックスできる場所でゆっくり過ごすのがおすすめです。"


def build_result_html(label, reason):
    """予測結果のカードの中身（HTML）を作る。"""
    emoji, name = CATEGORY_LABELS[label]
    chips = "".join(
        f'<span class="chip">{spot}</span>' for spot in CATEGORY_SPOTS[label]
    )
    return f"""
<div class="result-hero">
  <div class="result-emoji">{emoji}</div>
  <div class="result-text">
    <span class="result-caption">今日のおすすめ</span>
    <span class="result-name">{name}</span>
  </div>
</div>
<p class="result-reason">{reason}</p>
<p class="result-subtitle">こんな場所がおすすめ</p>
<div class="chip-list">{chips}</div>
"""


def recommend(temperature, rain_probability, wind_speed, humidity):
    """予測ボタンが押されたときに Gradio から呼ばれるメインの関数。"""
    label = predict_category(temperature, rain_probability, wind_speed, humidity)
    reason = build_reason(label, temperature, rain_probability, wind_speed, humidity)

    return (
        build_result_html(label, reason),   # 結果カードの中身
        label,                              # プラン作成のために覚えておく（gr.State）
        gr.update(visible=True),            # STEP 2 のカードを表示する
        gr.update(visible=True),            # STEP 3 のカードを表示する
    )


# ---------------------------------------------------------------
# お出かけプラン関連の関数
# ---------------------------------------------------------------

def resolve_location(mode, latitude_text, longitude_text, area_query):
    """プランを作る場所（緯度経度・地名）を決める。

    戻り値: (緯度, 経度, 地名, エラーメッセージ, お知らせ) のタプル。
            エラーメッセージ … これがあるときはプランを作れない（入力のミスなど）
            お知らせ … プランは作れるが、伝えておきたいこと（APIが使えなかった等）
    """
    if mode == LOCATION_MODE_GPS:
        if not latitude_text or not longitude_text:
            return None, None, None, (
                "⚠️ 「📍 現在地を取得」を押して、ブラウザの位置情報の利用を許可してください。"
            ), None
        try:
            latitude = float(latitude_text)
            longitude = float(longitude_text)
        except ValueError:
            return None, None, None, "⚠️ 現在地を正しく取得できませんでした。もう一度お試しください。", None

        # 緯度経度から地名を調べる（Google → OpenStreetMap の順に試す）
        area_name = (
            maps_api.reverse_geocode(latitude, longitude)
            or osm_api.reverse_geocode(latitude, longitude)
            or "現在地周辺"
        )
        return latitude, longitude, area_name, None, None

    # ここから下は「市区町村・都道府県から」の場合
    query = (area_query or "").strip()
    if not query:
        return None, None, None, "⚠️ 市区町村・都道府県を入力してください（例：京都市）。", None

    # 1. Google Maps で地名を緯度経度に変換する
    if maps_api.has_api_key() and maps_api.denied_reason() is None:
        try:
            latitude, longitude, area_name = maps_api.geocode(query)
            return latitude, longitude, area_name, None, None
        except maps_api.MapsError:
            pass

    # 2. だめなら OpenStreetMap で変換する
    try:
        latitude, longitude, area_name = osm_api.geocode(query)
        return latitude, longitude, area_name, None, None
    except osm_api.OsmError as error:
        return None, None, None, f"⚠️ その地名が見つかりませんでした（{error}）。", None


def make_plan(label, mode, latitude_text, longitude_text, area_query,
              wish_query, start_time, end_time, radius_km):
    """プラン作成ボタンが押されたときに Gradio から呼ばれる関数。"""
    if not label:
        return "⚠️ 先に STEP 1 の「おすすめを予測する」を押してください。", gr.update(visible=True)

    latitude, longitude, area_name, error, note = resolve_location(
        mode, latitude_text, longitude_text, area_query
    )
    if error:
        return error, gr.update(visible=True)

    plan_text = planner.build_plan(
        label,
        latitude,
        longitude,
        area_name,
        start_time,
        end_time,
        radius_m=int(radius_km * 1000),
        note=note,
        wishes=planner.parse_wishes(wish_query),
    )
    return plan_text, gr.update(visible=True)


def toggle_location_inputs(mode):
    """場所の決め方（現在地／地名）に合わせて、入力欄の表示を切り替える。"""
    is_gps = mode == LOCATION_MODE_GPS
    return (
        gr.update(visible=is_gps),       # 現在地取得ボタン
        gr.update(visible=is_gps),       # 現在地の表示欄
        gr.update(visible=not is_gps),   # 地名の入力欄
    )


# ---------------------------------------------------------------
# 画面（UI）
# ---------------------------------------------------------------

def step_head(step, title, description="", tone="s1"):
    """カードの見出し（STEPバッジ＋タイトル＋説明）のHTMLを作る。

    tone は STEPバッジの色（s1=オレンジ / s2=水色 / s3=みどり）。
    """
    desc_html = f'<p class="step-desc">{description}</p>' if description else ""
    return (
        f'<div class="step-head">'
        f'<span class="step-badge {tone}">{step}</span>'
        f'<span class="step-title">{title}</span>'
        f"</div>{desc_html}"
    )


# 起動時に1回だけ、APIキーが実際に使えるかを確かめる
MAPS_READY, MAPS_MESSAGE = maps_api.check_status()


def maps_status_pill():
    """Google Maps 連携の状態を示すバッジのHTMLを作る。"""
    if MAPS_READY:
        return '<span class="maps-pill on">🟢 Google Maps 連携中</span>'
    if not maps_api.has_api_key():
        return '<span class="maps-pill off">⚪ オフラインモード（検索リンクを表示）</span>'
    return f'<span class="maps-pill warn">🟡 {MAPS_MESSAGE}</span>'


# 空と太陽をイメージした、やわらかくて動きのあるデザイン
CSS = """
:root {
    --card: #ffffff;
    --text: #2b2b33;
    --muted: #7d8598;
    --accent: #ff8a3d;         /* メインのオレンジ（太陽） */
    --accent-dark: #f4701f;
    --accent-soft: #fff1e5;
    --sky: #3ba9ff;            /* サブの水色（空） */
    --sky-soft: #e8f4ff;
    --mint: #22a98a;
    --mint-soft: #e2f7f1;
    --border: #f1e8e0;
    --radius: 22px;
}

/* --- 全体 --- */
body,
gradio-app {
    background: linear-gradient(170deg, #fff6ec 0%, #fdf3ff 45%, #e9f4ff 100%) !important;
    background-attachment: fixed !important;
}

.gradio-container {
    position: relative;
    z-index: 1;
    background: transparent !important;
    width: min(100%, 720px) !important;
    max-width: 720px !important;
    margin: 0 auto !important;
    padding: 8px 16px 48px !important;
    box-sizing: border-box !important;
    color: var(--text) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Hiragino Kaku Gothic ProN",
        "Yu Gothic", Meiryo, sans-serif !important;
}

/* Gradio のフッター（Use via API など）は隠す */
footer {
    display: none !important;
}

/* --- 背景でゆっくり動く、ぼんやりした丸 --- */
#bg-decor {
    position: fixed;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
    z-index: -1;
}

#bg-decor span {
    position: absolute;
    display: block;
    border-radius: 50%;
    filter: blur(46px);
    opacity: 0.5;
    animation: blob 22s ease-in-out infinite;
}

#bg-decor .b1 {
    width: 260px; height: 260px; top: -60px; left: -60px;
    background: #ffd8a8;
}

#bg-decor .b2 {
    width: 300px; height: 300px; top: 30%; right: -90px;
    background: #bfe3ff; animation-delay: -7s;
}

#bg-decor .b3 {
    width: 240px; height: 240px; bottom: -70px; left: 10%;
    background: #ffd6e7; animation-delay: -14s;
}

/* --- ヘッダー --- */
#app-header {
    text-align: center;
    padding: 26px 0 20px;
}

#app-header .app-logo {
    font-size: 2.6rem;
    line-height: 1;
    display: inline-block;
    animation: floatY 3.6s ease-in-out infinite;
    filter: drop-shadow(0 6px 14px rgba(255, 170, 80, 0.45));
}

#app-header .app-title {
    margin: 10px 0 0;
    font-size: 1.72rem;
    font-weight: 800;
    letter-spacing: 0.01em;
    background: linear-gradient(92deg, var(--accent-dark) 0%, #ff9a4d 45%, var(--sky) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
}

#app-header .app-sub {
    margin: 10px 0 0;
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.75;
}

/* --- カード --- */
.card {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 22px !important;
    margin-bottom: 18px !important;
    box-shadow: 0 2px 4px rgba(120, 90, 60, 0.04), 0 14px 34px rgba(120, 90, 60, 0.09);
    box-sizing: border-box;
    animation: cardIn 0.55s cubic-bezier(0.21, 1.02, 0.73, 1) both;
}

/* --- ステップの見出し --- */
.step-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}

.step-badge {
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    padding: 5px 11px;
    border-radius: 999px;
    white-space: nowrap;
}

.step-badge.s1 { background: var(--accent-soft); color: var(--accent-dark); }
.step-badge.s2 { background: var(--sky-soft); color: #1f7fd0; }
.step-badge.s3 { background: var(--mint-soft); color: var(--mint); }

.step-title {
    font-size: 1.08rem;
    font-weight: 700;
}

.step-desc {
    margin: 0 0 14px;
    color: var(--muted);
    font-size: 0.86rem;
    line-height: 1.6;
}

/* --- 入力まわり --- */
.weather-row {
    gap: 16px !important;
}

.card label,
.card .label-wrap span {
    font-weight: 600 !important;
}

input[type="range"] {
    accent-color: var(--accent) !important;
}

/* ボタン */
.card button {
    border-radius: 999px !important;
    font-weight: 700 !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
}

.primary-btn,
.primary-btn button {
    background: linear-gradient(135deg, #ffa963 0%, #ff7a2f 100%) !important;
    border: none !important;
    color: #ffffff !important;
    min-height: 52px !important;
    font-size: 1.02rem !important;
    width: 100% !important;
    margin-top: 8px !important;
    box-shadow: 0 10px 20px rgba(255, 122, 47, 0.28) !important;
}

.primary-btn:hover,
.primary-btn button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 16px 28px rgba(255, 122, 47, 0.36) !important;
}

.primary-btn:active,
.primary-btn button:active {
    transform: translateY(0) scale(0.99) !important;
}

.sub-btn,
.sub-btn button {
    background: var(--sky-soft) !important;
    border: 1px solid #cfe7fb !important;
    color: #1f7fd0 !important;
    min-height: 46px !important;
}

.sub-btn:hover,
.sub-btn button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 18px rgba(59, 169, 255, 0.22) !important;
}

/* 場所の決め方（ラジオ） */
#location-mode label {
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    padding: 8px 14px !important;
    transition: background 0.18s ease, border-color 0.18s ease !important;
}

#location-mode label:hover {
    background: var(--accent-soft) !important;
    border-color: #ffd9bb !important;
}

/* Google Maps 連携の状態バッジ */
.maps-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.76rem;
    font-weight: 700;
    padding: 5px 12px;
    border-radius: 999px;
    margin-bottom: 12px;
}

.maps-pill.on { background: var(--mint-soft); color: var(--mint); }
.maps-pill.off { background: #f3f4f6; color: #7d8598; }
.maps-pill.warn { background: #fff4e0; color: #b06a12; }

/* --- STEP 2：予測結果 --- */
.result-hero {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-top: 12px;
}

.result-emoji {
    flex: none;
    width: 68px;
    height: 68px;
    border-radius: 24px;
    background: linear-gradient(140deg, #fff3e6 0%, #ffe6f0 100%);
    box-shadow: inset 0 0 0 1px #ffe3cd, 0 8px 18px rgba(255, 160, 90, 0.22);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.1rem;
    animation: popIn 0.6s cubic-bezier(0.18, 1.3, 0.5, 1) both;
}

.result-emoji:hover {
    animation: wiggle 0.6s ease;
}

.result-caption {
    display: block;
    color: var(--muted);
    font-size: 0.78rem;
    letter-spacing: 0.04em;
}

.result-name {
    display: block;
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1.3;
}

.result-reason {
    margin: 18px 0;
    padding: 14px 16px;
    background: linear-gradient(135deg, #fffaf4 0%, #f7fbff 100%);
    border-left: 4px solid var(--accent);
    border-radius: 4px 14px 14px 4px;
    color: #4a5160;
    font-size: 0.93rem;
    line-height: 1.8;
}

.result-subtitle {
    margin: 0 0 10px;
    color: var(--muted);
    font-size: 0.8rem;
    font-weight: 700;
}

.chip-list {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.chip {
    background: var(--sky-soft);
    color: #2b7fc0;
    border-radius: 999px;
    padding: 7px 14px;
    font-size: 0.85rem;
    font-weight: 600;
    animation: chipIn 0.45s ease both;
    transition: transform 0.16s ease;
}

.chip:hover { transform: translateY(-2px) rotate(-1.5deg); }

.chip:nth-child(1) { animation-delay: 0.10s; }
.chip:nth-child(2) { animation-delay: 0.18s; }
.chip:nth-child(3) { animation-delay: 0.26s; }
.chip:nth-child(4) { animation-delay: 0.34s; }

/* --- STEP 3：プランの表 --- */
#plan-output {
    overflow-x: auto;
    margin-top: 2px;
}

#plan-output h3 {
    font-size: 1.08rem !important;
    margin-bottom: 4px !important;
}

#plan-output table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.9rem;
    margin-top: 12px;
}

#plan-output th {
    background: #fffaf4;
    color: var(--muted);
    font-weight: 700;
    text-align: left;
    padding: 10px;
    white-space: nowrap;
}

#plan-output td {
    padding: 13px 10px;
    border-top: 1px solid var(--border);
    vertical-align: top;
}

#plan-output tbody tr {
    animation: rowIn 0.45s ease both;
}

#plan-output tbody tr:nth-child(1) { animation-delay: 0.05s; }
#plan-output tbody tr:nth-child(2) { animation-delay: 0.13s; }
#plan-output tbody tr:nth-child(3) { animation-delay: 0.21s; }
#plan-output tbody tr:nth-child(4) { animation-delay: 0.29s; }
#plan-output tbody tr:nth-child(5) { animation-delay: 0.37s; }
#plan-output tbody tr:nth-child(6) { animation-delay: 0.45s; }

#plan-output tbody tr:hover {
    background: #fffaf4;
}

#plan-output td:first-child {
    white-space: nowrap;
    color: var(--accent-dark);
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}

#plan-output a {
    color: #1f7fd0;
    font-weight: 700;
    text-decoration: none;
    border-bottom: 2px solid #d6ecff;
}

#plan-output a:hover {
    border-bottom-color: #1f7fd0;
}

/* --- フッター --- */
#app-footer {
    text-align: center;
    color: #a6afbd;
    font-size: 0.8rem;
    line-height: 1.8;
    padding-top: 4px;
}

/* --- アニメーション --- */
@keyframes floatY {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-9px); }
}

@keyframes cardIn {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: none; }
}

@keyframes popIn {
    0% { opacity: 0; transform: scale(0.4) rotate(-12deg); }
    70% { opacity: 1; transform: scale(1.12) rotate(4deg); }
    100% { opacity: 1; transform: scale(1) rotate(0); }
}

@keyframes wiggle {
    0%, 100% { transform: rotate(0); }
    25% { transform: rotate(-8deg); }
    75% { transform: rotate(8deg); }
}

@keyframes chipIn {
    from { opacity: 0; transform: translateY(8px) scale(0.94); }
    to { opacity: 1; transform: none; }
}

@keyframes rowIn {
    from { opacity: 0; transform: translateX(-10px); }
    to { opacity: 1; transform: none; }
}

@keyframes blob {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(24px, -28px) scale(1.08); }
    66% { transform: translate(-22px, 18px) scale(0.94); }
}

/* アニメーションが苦手な人の設定を尊重する */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
    }
}

/* --- スマホ向け --- */
@media (max-width: 560px) {
    .gradio-container {
        padding: 4px 12px 40px !important;
    }

    #app-header {
        padding: 18px 0 14px;
    }

    #app-header .app-logo { font-size: 2.2rem; }
    #app-header .app-title { font-size: 1.45rem; }
    #app-header .app-sub { font-size: 0.86rem; }

    .card {
        padding: 17px !important;
        border-radius: 18px !important;
        margin-bottom: 14px !important;
    }

    /* 横並びのスライダーを縦積みにする */
    .weather-row {
        flex-direction: column !important;
        gap: 10px !important;
    }

    .weather-row > * {
        width: 100% !important;
        min-width: 0 !important;
    }

    .card button {
        width: 100% !important;
        min-height: 48px !important;
    }

    .result-emoji {
        width: 58px;
        height: 58px;
        border-radius: 20px;
        font-size: 1.8rem;
    }

    .result-name { font-size: 1.35rem; }

    #plan-output table { font-size: 0.82rem; }

    /* 結果表示エリアの文字を読みやすくする */
    .card textarea,
    .card input {
        font-size: 16px !important;
    }
}
"""


def create_app():
    """Gradio の画面を組み立てて返す。"""
    with gr.Blocks(css=CSS, theme="soft", title="お出かけプランナー") as app:

        # 予測したカテゴリを覚えておく場所（画面には表示されない）
        predicted_label = gr.State(value="")

        # --- ヘッダー ---
        gr.HTML(
            """
<div id="bg-decor"><span class="b1"></span><span class="b2"></span><span class="b3"></span></div>
<div id="app-header">
  <div class="app-logo">☀️</div>
  <h1 class="app-title">お出かけプランナー</h1>
  <p class="app-sub">天気からAIがおすすめを予測して、<br>行き先と時間まで決めたプランを作ります。</p>
</div>
""",
        )

        # --- STEP 1：天気の入力 ---
        with gr.Column(elem_classes="card", elem_id="step-weather"):
            gr.HTML(step_head(
                "STEP 1", "☁️ 今日の天気",
                "4つの数値を動かして、今日の天気に合わせてください。",
                tone="s1",
            ))

            with gr.Row(elem_classes="weather-row"):
                temperature = gr.Slider(
                    minimum=-10, maximum=40, value=22, step=0.5, label="気温（℃）"
                )
                rain_probability = gr.Slider(
                    minimum=0, maximum=100, value=20, step=5, label="降水確率（%）"
                )

            with gr.Row(elem_classes="weather-row"):
                wind_speed = gr.Slider(
                    minimum=0, maximum=20, value=3, step=0.5, label="風速（m/s）"
                )
                humidity = gr.Slider(
                    minimum=0, maximum=100, value=50, step=5, label="湿度（%）"
                )

            predict_button = gr.Button(
                "おすすめを予測する", variant="primary", size="lg",
                elem_classes="primary-btn",
            )

        # --- STEP 2：予測結果（予測するまでは表示しない） ---
        with gr.Column(elem_classes="card", elem_id="step-result", visible=False) as result_card:
            gr.HTML(step_head("STEP 2", "✨ AIの予測結果", tone="s2"))
            result_html = gr.HTML()

        # --- STEP 3：お出かけプラン（予測するまでは表示しない） ---
        with gr.Column(elem_classes="card", elem_id="step-plan", visible=False) as plan_card:
            gr.HTML(step_head(
                "STEP 3", "🗺️ お出かけプランを作る",
                "場所と時間を決めると、その時間内でまわれるプランを作ります。",
                tone="s3",
            ))
            gr.HTML(maps_status_pill())

            location_mode = gr.Radio(
                choices=[LOCATION_MODE_GPS, LOCATION_MODE_AREA],
                value=LOCATION_MODE_GPS,
                label="どこのプランを作る？",
                elem_id="location-mode",
            )

            # 「現在地から」のときだけ表示する
            locate_button = gr.Button("📍 現在地を取得", elem_classes="sub-btn")
            location_status = gr.Textbox(
                label="現在地",
                value="まだ取得していません。",
                interactive=False,
            )

            # 「市区町村・都道府県から」のときだけ表示する
            area_query = gr.Textbox(
                label="市区町村・都道府県",
                placeholder="例：京都市 / 神奈川県横浜市 / 沖縄県",
                visible=False,
            )

            wish_query = gr.Textbox(
                label="行きたい場所（任意・4つまで）",
                placeholder="例：はま寿司、水族館、スターバックス",
                info="お店や施設の名前を、読点（、）で区切って入力すると、プランの前のほうに入れます。",
            )

            with gr.Row(elem_classes="weather-row"):
                start_time = gr.Dropdown(
                    choices=planner.time_choices(), value="10:00", label="開始時刻"
                )
                end_time = gr.Dropdown(
                    choices=planner.time_choices(), value="17:00", label="終了時刻"
                )

            radius_km = gr.Slider(
                minimum=1, maximum=20, value=3, step=1, label="スポットを探す範囲（km）"
            )

            plan_button = gr.Button(
                "プランを作る", variant="primary", size="lg", elem_classes="primary-btn"
            )

            # 現在地の緯度経度を保存しておく欄（画面には表示しない）
            latitude_box = gr.Textbox(visible=False)
            longitude_box = gr.Textbox(visible=False)

        # --- プランの表示エリア（プランを作るまでは表示しない） ---
        with gr.Column(elem_classes="card", elem_id="step-plan-result", visible=False) as plan_result_card:
            plan_output = gr.Markdown(elem_id="plan-output")

        gr.HTML(
            '<p id="app-footer">☀️ 🌦️ 🌈 &nbsp; 授業用のサンプルアプリです &nbsp; 🌈 🌦️ ☀️<br>'
            'スポット情報：Google Maps</p>'
        )

        # --- ボタンと関数をつなぐ ---
        predict_button.click(
            fn=recommend,
            inputs=[temperature, rain_probability, wind_speed, humidity],
            outputs=[result_html, predicted_label, result_card, plan_card],
        ).then(
            fn=None, inputs=None, outputs=None, js=scroll_to_js("step-result")
        )

        location_mode.change(
            fn=toggle_location_inputs,
            inputs=location_mode,
            outputs=[locate_button, location_status, area_query],
        )

        # ブラウザの中だけで動く処理なので、fn は使わず js だけを指定する
        locate_button.click(
            fn=None,
            inputs=None,
            outputs=[latitude_box, longitude_box, location_status],
            js=GET_LOCATION_JS,
        )

        plan_button.click(
            fn=make_plan,
            inputs=[
                predicted_label,
                location_mode,
                latitude_box,
                longitude_box,
                area_query,
                wish_query,
                start_time,
                end_time,
                radius_km,
            ],
            outputs=[plan_output, plan_result_card],
        ).then(
            fn=None, inputs=None, outputs=None, js=scroll_to_js("step-plan-result")
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
