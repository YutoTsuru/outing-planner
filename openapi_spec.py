"""REST API の仕様（OpenAPI 3.0）を組み立てる。

api.py のルート定義を実行時に読み取って自動生成するのではなく、
ここに**手で書いた**仕様を1か所にまとめています。

自動生成にしなかった理由は、Flask のルートからは「どんな入力を受け付けるか」
「エラー時に何を返すか」までは読み取れないためです。無理に自動化すると、
実際とは違う仕様書ができて、かえって信用できないものになります。
そのかわり、api.py の各エンドポイントを変えたときは、このファイルも
あわせて直すことを忘れないでください（tests/test_openapi.py が
パスの一覧だけは機械的に突き合わせています）。
"""

from outing_ml.config import CATEGORIES, FEATURE_COLUMNS

# ドキュメント上に出す入力例（doc/webapp.md の例と揃えている）
_WEATHER_EXAMPLE = {
    "temperature": 22, "rain_probability": 10, "wind_speed": 2, "humidity": 50,
}

_OK_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}

_ERROR_RESPONSE = {
    "description": "入力の誤りや、内部の問題。",
    "content": {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "ok": {"type": "boolean", "example": False},
                    "error": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                            "details": {"type": "object"},
                        },
                        "required": ["message"],
                    },
                },
                "required": ["ok", "error"],
            }
        }
    },
}

_WEATHER_SCHEMA = {
    "type": "object",
    "properties": {column: {"type": "number"} for column in FEATURE_COLUMNS},
    "required": FEATURE_COLUMNS,
}

_RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "probabilities": {"type": "object"},
        "confidence": {"type": "number"},
        "comfort_score": {"type": "number", "nullable": True},
        "weather_type": {"type": "integer", "nullable": True},
        "weather_type_name": {"type": "string", "nullable": True},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "model_versions": {"type": "object"},
    },
}

_CITY_PARAM = {
    "name": "city",
    "in": "query",
    "required": True,
    "schema": {"type": "string"},
    "description": "都市名（`/api/cities` で一覧を取れる）。学習していない都市は 400。",
    "example": "東京",
}


def _json_response(description: str, schema: dict, example: dict = None) -> dict:
    content = {"schema": schema}
    if example is not None:
        content["example"] = example
    return {"description": description, "content": {"application/json": content}}


def build_spec() -> dict:
    """OpenAPI 3.0 の仕様（辞書。そのまま JSON にできる）を返す。"""
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "お出かけプランナー API",
            "description": (
                "天気からお出かけカテゴリ・おでかけ日和度・天気タイプを予測し、"
                "翌日以降の予報やお出かけプランを作る REST API。"
                "モデルの詳しい説明は doc/README.md、使い方は doc/webapp.md を参照。"
            ),
            "version": "1.0.0",
        },
        "servers": [{"url": "/api", "description": "このサーバの /api 以下"}],
        "tags": [
            {"name": "状態", "description": "モデルの読み込み状況・履歴・監視"},
            {"name": "予測", "description": "天気からおすすめを予測する"},
            {"name": "予報", "description": "翌日・週間の天気予報"},
            {"name": "プラン", "description": "お出かけプランの作成"},
        ],
        "paths": {
            "/health": {
                "get": {
                    "tags": ["状態"],
                    "summary": "読み込めているモデルと版を確認する",
                    "responses": {"200": _json_response(
                        "読み込めているモデルの情報。", _OK_SCHEMA,
                        {"ok": True, "categories": CATEGORIES,
                         "features": FEATURE_COLUMNS,
                         "models": {"category": "v2", "comfort": "v2",
                                   "weather_type": "v2"}, "forecast": "v2"},
                    )},
                }
            },
            "/models": {
                "get": {
                    "tags": ["状態"],
                    "summary": "学習の履歴を見る（版・成績・データの指紋）",
                    "responses": {"200": _json_response(
                        "モデルごとの最新の学習記録。", _OK_SCHEMA)},
                }
            },
            "/cities": {
                "get": {
                    "tags": ["予報"],
                    "summary": "翌日予報を出せる都市の一覧",
                    "responses": {"200": _json_response(
                        "都市名の配列（全都道府県庁所在地・47件）。", _OK_SCHEMA,
                        {"ok": True, "cities": ["札幌", "仙台", "東京"]})},
                }
            },
            "/weather-types": {
                "get": {
                    "tags": ["状態"],
                    "summary": "天気タイプ分けモデルが作った、天気タイプの一覧",
                    "responses": {
                        "200": _json_response("天気タイプの一覧。", _OK_SCHEMA),
                        "503": _ERROR_RESPONSE,
                    },
                }
            },
            "/predict": {
                "post": {
                    "tags": ["予測"],
                    "summary": "天気4項目からおすすめを予測する",
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {
                            "schema": _WEATHER_SCHEMA, "example": _WEATHER_EXAMPLE,
                        }},
                    },
                    "responses": {
                        "200": _json_response(
                            "予測結果。", {"allOf": [_OK_SCHEMA, _RECOMMENDATION_SCHEMA]}),
                        "400": _ERROR_RESPONSE,
                    },
                }
            },
            "/predict/batch": {
                "post": {
                    "tags": ["予測"],
                    "summary": "天気をまとめて予測する（最大100件）",
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "days": {"type": "array", "items": _WEATHER_SCHEMA},
                                },
                                "required": ["days"],
                            },
                            "example": {"days": [_WEATHER_EXAMPLE]},
                        }},
                    },
                    "responses": {
                        "200": _json_response("予測結果の配列。", _OK_SCHEMA),
                        "400": _ERROR_RESPONSE,
                    },
                }
            },
            "/forecast": {
                "get": {
                    "tags": ["予報"],
                    "summary": "あしたの天気とおすすめを予測する",
                    "parameters": [
                        _CITY_PARAM,
                        {"name": "days", "in": "query", "required": False,
                         "schema": {"type": "integer", "minimum": 5, "maximum": 30,
                                   "default": 12},
                         "description": "予測の材料に使う、直近何日ぶんの実測を取得するか。"},
                    ],
                    "responses": {
                        "200": _json_response("あしたの天気と予測区間、おすすめ。", _OK_SCHEMA),
                        "400": _ERROR_RESPONSE,
                        "503": _ERROR_RESPONSE,
                    },
                }
            },
            "/week": {
                "get": {
                    "tags": ["予報"],
                    "summary": "数日先まで（最大7日）の天気とおすすめを予測する（再帰予測）",
                    "parameters": [
                        _CITY_PARAM,
                        {"name": "days", "in": "query", "required": False,
                         "schema": {"type": "integer", "minimum": 1, "maximum": 7,
                                   "default": 7},
                         "description": "何日先まで出すか。2日目以降は予測を入力に使い直す"
                                       "再帰予測のため、日を追うごとに精度は未検証で下がる。"},
                        {"name": "history_days", "in": "query", "required": False,
                         "schema": {"type": "integer", "minimum": 5, "maximum": 30,
                                   "default": 10}},
                    ],
                    "responses": {
                        "200": _json_response("日ごとの天気・予測区間・おすすめの配列。",
                                              _OK_SCHEMA),
                        "400": _ERROR_RESPONSE,
                        "503": _ERROR_RESPONSE,
                    },
                }
            },
            "/compare": {
                "get": {
                    "tags": ["予報"],
                    "summary": "全都市のあしたの日和度を高い順に比較する",
                    "parameters": [
                        {"name": "refresh", "in": "query", "required": False,
                         "schema": {"type": "string", "enum": ["1"]},
                         "description": "'1' を指定すると、30分キャッシュを無視して取得し直す"
                                       "（初回は47都市ぶんの取得で数分かかる）。"},
                    ],
                    "responses": {
                        "200": _json_response("日和度が高い順の都市一覧。", _OK_SCHEMA),
                        "503": _ERROR_RESPONSE,
                    },
                }
            },
            "/monitor": {
                "get": {
                    "tags": ["状態"],
                    "summary": "学習データと最近の入力のずれを見る（ドリフト監視）",
                    "responses": {
                        "200": _json_response(
                            "PSI・KS統計量とOK/WATCH/ALERT判定。", _OK_SCHEMA),
                        "503": _ERROR_RESPONSE,
                    },
                }
            },
            "/history": {
                "get": {
                    "tags": ["状態"],
                    "summary": "これまでの予測の記録と、その傾向を見る",
                    "parameters": [
                        {"name": "limit", "in": "query", "required": False,
                         "schema": {"type": "integer", "minimum": 1, "maximum": 1000,
                                   "default": 100}},
                    ],
                    "responses": {
                        "200": _json_response("記録の一覧と集計。", _OK_SCHEMA),
                        "400": _ERROR_RESPONSE,
                    },
                }
            },
            "/plan": {
                "post": {
                    "tags": ["プラン"],
                    "summary": "時間つきのお出かけプランを作る",
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "category": {"type": "string", "enum": CATEGORIES},
                                    "area": {"type": "string",
                                            "description": "市区町村・都道府県名。"
                                                          "latitude/longitude の指定でもよい。"},
                                    "latitude": {"type": "number"},
                                    "longitude": {"type": "number"},
                                    "start_time": {"type": "string", "example": "10:00"},
                                    "end_time": {"type": "string", "example": "17:00"},
                                    "radius_km": {"type": "integer", "minimum": 1,
                                                 "maximum": 20, "default": 3},
                                    "wishes": {"type": "array", "items": {"type": "string"},
                                              "description": "行きたい場所の名前（最大4件）。"},
                                },
                                "required": ["category"],
                            },
                            "example": {"category": "outdoor", "area": "京都市",
                                       "start_time": "10:00", "end_time": "15:00"},
                        }},
                    },
                    "responses": {
                        "200": _json_response(
                            "Markdown形式のプランと、共有リンク（share_url）。",
                            _OK_SCHEMA,
                            {"ok": True, "category": "outdoor", "area": "京都府京都市",
                             "plan_markdown": "### 🗺️ 京都府京都市 のお出かけプラン\n...",
                             "share_id": "608e7cadbd08", "share_url": "/share/608e7cadbd08"},
                        ),
                        "400": _ERROR_RESPONSE,
                        "404": _ERROR_RESPONSE,
                    },
                }
            },
            "/share/{plan_id}": {
                "get": {
                    "tags": ["プラン"],
                    "summary": "共有リンクからプランを取り出す",
                    "description": (
                        "有効期限やアクセス制限は無い。IDを知っている人なら誰でも開ける。"
                    ),
                    "parameters": [
                        {"name": "plan_id", "in": "path", "required": True,
                         "schema": {"type": "string"},
                         "description": "/api/plan のレスポンスに入っている share_id。"},
                    ],
                    "responses": {
                        "200": _json_response("保存されたプラン。", _OK_SCHEMA),
                        "404": _ERROR_RESPONSE,
                    },
                }
            },
        },
    }
