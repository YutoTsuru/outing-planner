"""画面に出す文言と、入力欄の決まりごと。

画面が2つ（Gradio と Flask）あるので、カテゴリの表示名・おすすめ理由・入力欄の範囲を
ここ1か所に置きます。2か所に書くと、片方だけ直して食い違うためです。
"""

from outing_ml.config import CATEGORIES, INPUT_RANGES

# カテゴリの (アイコン, 表示名)
CATEGORY_LABELS = {
    "outdoor": ("🌳", "屋外観光"),
    "indoor": ("🏛️", "屋内観光"),
    "relax": ("☕", "リラックス"),
}

# カテゴリごとのおすすめスポット例
CATEGORY_SPOTS = {
    "outdoor": ["公園", "神社", "海辺", "動物園"],
    "indoor": ["水族館", "美術館", "博物館", "映画館"],
    "relax": ["カフェ", "温泉", "図書館", "スパ"],
}

# 入力欄の見た目（範囲は outing_ml.config の INPUT_RANGES が正本）
WEATHER_FIELDS = [
    {"name": "temperature", "label": "気温", "unit": "℃", "step": 0.5, "value": 22},
    {"name": "rain_probability", "label": "降水確率", "unit": "%", "step": 5, "value": 20},
    {"name": "wind_speed", "label": "風速", "unit": "m/s", "step": 0.5, "value": 3},
    {"name": "humidity", "label": "湿度", "unit": "%", "step": 5, "value": 50},
]


def weather_fields():
    """入力欄の定義に、設定側の範囲を足して返す。"""
    fields = []
    for field in WEATHER_FIELDS:
        low, high = INPUT_RANGES[field["name"]]
        fields.append({**field, "min": low, "max": high})
    return fields


def label_view(category: str) -> dict:
    """カテゴリ1つぶんの表示用データ。"""
    emoji, name = CATEGORY_LABELS[category]
    return {"key": category, "emoji": emoji, "name": name,
            "spots": CATEGORY_SPOTS[category]}


def label_views() -> dict:
    """全カテゴリの表示用データ。"""
    return {category: label_view(category) for category in CATEGORIES}


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
