"""
お出かけプランナー：Gradioアプリ

気象データ（気温・降水確率・風速・湿度）を入力すると、
学習済みモデルがおすすめのお出かけカテゴリを予測して表示します。

起動前に train_model.py を実行して、モデルを作成しておいてください。

実行方法:
    python app.py
"""

import os

import gradio as gr
import joblib
import pandas as pd

# ---------------------------------------------------------------
# 設定・固定データ
# ---------------------------------------------------------------

# モデルの場所（train_model.py と合わせる）
MODEL_PATH = os.path.join("model", "outing_model.pkl")

# 特徴量の列名（train_model.py と同じ順番にすること）
FEATURE_COLUMNS = ["temperature", "rain_probability", "wind_speed", "humidity"]

# カテゴリの表示名
CATEGORY_NAMES = {
    "outdoor": "🌳 屋外観光",
    "indoor": "🏛️ 屋内観光",
    "relax": "☕ リラックス",
}

# カテゴリごとのおすすめスポット例（今回は固定データ）
CATEGORY_SPOTS = {
    "outdoor": ["公園", "神社", "海辺", "動物園"],
    "indoor": ["水族館", "美術館", "博物館", "映画館"],
    "relax": ["カフェ", "温泉", "図書館", "スパ"],
}


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


def recommend(temperature, rain_probability, wind_speed, humidity):
    """予測ボタンが押されたときに Gradio から呼ばれるメインの関数。"""
    label = predict_category(temperature, rain_probability, wind_speed, humidity)

    category_name = CATEGORY_NAMES[label]
    reason = build_reason(label, temperature, rain_probability, wind_speed, humidity)
    spots = "、".join(CATEGORY_SPOTS[label])

    return category_name, reason, spots


# ---------------------------------------------------------------
# 画面（UI）
# ---------------------------------------------------------------

# 空・雲をイメージした、淡い青系のデザイン
CSS = """
/* 全体 */
.gradio-container {
    background: linear-gradient(180deg, #c9e6ff 0%, #e6f3ff 55%, #ffffff 100%) !important;
    width: min(100%, 860px) !important;
    max-width: 860px !important;
    margin: 0 auto !important;
    padding: 24px !important;
    box-sizing: border-box !important;
}

/* タイトルと説明文 */
#app-title {
    text-align: center;
    margin-top: 8px;
}

#app-description {
    text-align: center;
    color: #46698c;
    margin-bottom: 20px;
}

/* カード型のエリア（入力・結果で共通） */
.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 20px !important;
    box-shadow: 0 4px 14px rgba(90, 150, 210, 0.18);
    margin-bottom: 16px;
    box-sizing: border-box;
}

/* 入力行 */
.weather-row {
    gap: 16px !important;
}

/* フッターの注意書き */
#app-footer {
    text-align: center;
    color: #7d9ab5;
    font-size: 0.85em;
}

/* スマホ向け */
@media (max-width: 560px) {
    .gradio-container {
        padding: 12px !important;
    }

    #app-title h1 {
        font-size: 1.55rem !important;
        line-height: 1.35 !important;
    }

    #app-description {
        font-size: 0.92rem;
        line-height: 1.6;
        margin-bottom: 16px;
    }

    .card {
        padding: 14px !important;
        border-radius: 12px;
        margin-bottom: 12px;
    }

    .card h3 {
        font-size: 1.05rem !important;
    }

    /* 横並びのスライダーを縦積みにする */
    .weather-row {
        flex-direction: column !important;
        gap: 8px !important;
    }

    .weather-row > * {
        width: 100% !important;
        min-width: 0 !important;
    }

    /* ボタンを押しやすい高さ・横幅にする */
    .card .gr-button {
        width: 100% !important;
        min-height: 48px !important;
        font-size: 1rem !important;
    }

    /* 結果表示エリアの文字を読みやすくする */
    .card textarea,
    .card input {
        font-size: 16px !important;
    }
}

/* タブレット向け */
@media (min-width: 561px) and (max-width: 768px) {
    .gradio-container {
        padding: 20px !important;
    }

    .card {
        padding: 18px !important;
    }

    .weather-row {
        gap: 12px !important;
    }
}
"""


def create_app():
    """Gradio の画面を組み立てて返す。"""
    theme = gr.themes.Soft(primary_hue="sky")

    with gr.Blocks(css=CSS, theme=theme, title="お出かけプランナー") as app:
        # --- ヘッダー ---
        gr.Markdown("# ☀️ お出かけプランナー", elem_id="app-title")
        gr.Markdown(
            "今日の天気を入力すると、AIがぴったりのお出かけカテゴリを予測します。",
            elem_id="app-description",
        )

        # --- 入力エリア（カード） ---
        with gr.Column(elem_classes="card"):
            gr.Markdown("### 🌤️ 今日の天気を入力")
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
                "🔍 おすすめを予測する", variant="primary", size="lg"
            )

        # --- 結果エリア（カード） ---
        with gr.Column(elem_classes="card"):
            gr.Markdown("### 📋 予測結果")
            category_output = gr.Textbox(label="おすすめカテゴリ", interactive=False)
            reason_output = gr.Textbox(label="理由", interactive=False)
            spots_output = gr.Textbox(label="おすすめスポット例", interactive=False)

        gr.Markdown("※ 授業用のサンプルアプリです。", elem_id="app-footer")

        # --- ボタンと関数をつなぐ ---
        predict_button.click(
            fn=recommend,
            inputs=[temperature, rain_probability, wind_speed, humidity],
            outputs=[category_output, reason_output, spots_output],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
