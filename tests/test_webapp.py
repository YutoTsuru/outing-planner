"""Web アプリ（画面）のテスト。

外部への通信はしません（地名の変換とスポット検索は差し替えます）。
"""

import os

import pytest

from outing_ml.config import CONFIG
from outing_ml.serve import OutingService

MODELS_READY = os.path.exists(CONFIG.paths.category_model)
needs_models = pytest.mark.skipif(
    not MODELS_READY, reason="python train_all.py を先に実行してください"
)

GOOD_DAY = {"temperature": "22", "rain_probability": "10",
            "wind_speed": "2", "humidity": "50"}


class FakeForecast:
    """通信せずに、決まった予報を返す差し替え用。"""

    class _Bundle:
        version = "test"

    bundle = _Bundle()

    def __init__(self, outing):
        self.outing = outing

    def predict_tomorrow(self, city, days=12):
        from outing_ml.forecasting import TomorrowForecast

        weather = {"temperature": 22.0, "rain_probability": 10.0,
                   "wind_speed": 2.0, "humidity": 50.0}
        return TomorrowForecast(
            city=city, base_date="2026-08-27", target_date="2026-08-28",
            weather=weather,
            interval={"temperature": {"low": 20.0, "high": 24.0}},
            recommendation=self.outing.predict(**weather),
            model_version="test", notes=["テスト用の予報です"],
        )

    def predict_week(self, city, days_ahead=7, history_days=10):
        from outing_ml.forecasting import DailyForecast, WeeklyForecast

        weather = {"temperature": 22.0, "rain_probability": 10.0,
                   "wind_speed": 2.0, "humidity": 50.0}
        days = [
            DailyForecast(
                day=day, date=f"2026-08-{27 + day:02d}", weather=weather,
                interval={"temperature": {"low": 20.0 - day, "high": 24.0 + day}},
                recommendation=self.outing.predict(**weather),
            )
            for day in range(1, days_ahead + 1)
        ]
        return WeeklyForecast(city=city, base_date="2026-08-27", days=days,
                              model_version="test", notes=["週間予報のテスト用"])


@pytest.fixture
def client(tmp_path, monkeypatch):
    import prediction_log
    import webapp

    monkeypatch.setattr(prediction_log, "LOG_PATH", str(tmp_path / "predictions.jsonl"))

    outing = OutingService.load()
    app = webapp.create_web_app(outing=outing, forecast=FakeForecast(outing))
    app.config.update(TESTING=True)
    return app.test_client()


def text_of(response) -> str:
    return response.get_data(as_text=True)


@needs_models
def test_トップ画面に入力フォームが出る(client):
    body = text_of(client.get("/"))

    assert "今日の天気を入れる" in body
    assert 'name="temperature"' in body
    assert "東京" in body, "都市の選択肢が出る"


@needs_models
def test_予測すると結果が画面に出る(client):
    body = text_of(client.post("/predict", data=GOOD_DAY))

    assert "屋外観光" in body
    assert "おでかけ日和度" in body
    assert "今日の天気タイプ" in body
    assert "カテゴリごとの確率" in body


@needs_models
def test_数値でない入力はエラー画面になる(client):
    response = client.post("/predict", data={**GOOD_DAY, "temperature": "あつい"})

    assert response.status_code == 400
    assert "数値で入力してください" in text_of(response)


@needs_models
def test_入力が空ならエラー画面になる(client):
    response = client.post("/predict", data={**GOOD_DAY, "humidity": ""})
    assert response.status_code == 400


@needs_models
def test_範囲外の入力は警告つきで結果を出す(client):
    body = text_of(client.post("/predict", data={**GOOD_DAY, "temperature": "99"}))

    assert "屋外観光" in body or "リラックス" in body or "屋内観光" in body
    assert "範囲" in body, "丸めたことを画面で伝える"


@needs_models
def test_あしたの予報の画面が出る(client):
    body = text_of(client.get("/forecast?city=東京"))

    assert "2026-08-28 の予報" in body
    assert "だいたいこの範囲" in body
    assert "テスト用の予報です" in body


@needs_models
def test_都市の指定が無ければトップへ戻す(client):
    assert client.get("/forecast").status_code == 302


@needs_models
def test_モデルの状態が画面で分かる(client):
    body = text_of(client.get("/models"))

    assert "category-classifier" in body
    assert "学習の履歴" in body
    assert "指紋" in body


@needs_models
def test_無いページは404の画面になる(client):
    response = client.get("/nothing-here")

    assert response.status_code == 404
    assert "そのページはありません" in text_of(response)


@needs_models
def test_プランが作れる(client, monkeypatch):
    import geocoding
    import planner

    monkeypatch.setattr(geocoding, "resolve_area",
                        lambda query: (35.0, 135.6, "京都府京都市"))
    monkeypatch.setattr(planner, "build_plan",
                        lambda *args, **kwargs: "### テスト用のプラン")

    body = text_of(client.post("/plan", data={
        "category": "outdoor", "area": "京都市",
        "start_time": "10:00", "end_time": "15:00", "radius_km": "3",
    }))

    assert "京都府京都市" in body
    assert "テスト用のプラン" in body
    assert "/share/" in body


@needs_models
def test_プランの共有リンクから同じ内容を開ける(client, monkeypatch, tmp_path):
    import re

    import geocoding
    import planner
    import shared_plans

    # webapp.py は `import shared_plans` で同じモジュールを参照しているので、
    # ここで STORE_PATH を差し替えれば webapp 側にも反映される
    monkeypatch.setattr(shared_plans, "STORE_PATH", str(tmp_path / "shared_plans.jsonl"))
    monkeypatch.setattr(geocoding, "resolve_area",
                        lambda query: (35.0, 135.6, "京都府京都市"))
    monkeypatch.setattr(planner, "build_plan",
                        lambda *args, **kwargs: "### 共有用のプラン本文")

    created = client.post("/plan", data={
        "category": "outdoor", "area": "京都市",
        "start_time": "10:00", "end_time": "15:00", "radius_km": "3",
    })
    share_id = re.search(r"/share/([0-9a-f]{12})", text_of(created)).group(1)

    body = text_of(client.get(f"/share/{share_id}"))
    assert "共有用のプラン本文" in body
    assert "共有リンクから開いたプラン" in body


@needs_models
def test_存在しない共有リンクはエラー画面になる(client):
    response = client.get("/share/no-such-id")
    assert response.status_code == 404


@needs_models
def test_見つからない地名はエラー画面になる(client, monkeypatch):
    import geocoding

    def raise_not_found(query):
        raise geocoding.AreaNotFoundError(f"「{query}」が見つかりませんでした")

    monkeypatch.setattr(geocoding, "resolve_area", raise_not_found)

    response = client.post("/plan", data={"category": "outdoor", "area": "ありえない地名"})
    assert response.status_code == 404
    assert "見つかりません" in text_of(response)


@needs_models
def test_同じサーバでRESTAPIも動く(client):
    body = client.get("/api/health").get_json()
    assert body["ok"]

    result = client.post("/api/predict", json={"temperature": 22, "rain_probability": 10,
                                               "wind_speed": 2, "humidity": 50}).get_json()
    assert result["category"] == "outdoor"


@needs_models
def test_APIの無いURLはJSONで404を返す(client):
    response = client.get("/api/nothing")

    assert response.status_code == 404
    assert response.get_json()["ok"] is False


@needs_models
def test_記録がなければ案内を出す(client):
    body = text_of(client.get("/history"))

    assert "まだ記録がありません" in body


@needs_models
def test_予測すると記録の画面に出る(client):
    client.post("/predict", data=GOOD_DAY)

    body = text_of(client.get("/history"))
    assert "予測されたカテゴリの内訳" in body
    assert "使われたときの平均の天気" in body
    assert "web_predict" in body


@needs_models
def test_記録の件数指定が数値でなければ断る(client):
    assert client.get("/history?limit=たくさん").status_code == 400


@needs_models
def test_監視画面が表示される(client):
    body = text_of(client.get("/monitor"))

    assert "ドリフト監視" in body
    assert "総合判定" in body
    assert "天気4項目のずれ" in body


@needs_models
def test_都市比較の画面が表示される(client, monkeypatch):
    import webapp

    def fake_get_comparison(service, force_refresh=False):
        return {
            "rankings": [
                {"city": "那覇", "target_date": "2026-08-31",
                 "weather": {"temperature": 28.0, "rain_probability": 5.0,
                            "wind_speed": 3.0, "humidity": 60.0},
                 "category": "outdoor", "comfort_score": 90.0,
                 "weather_type_name": "過ごしやすい晴れの日", "confidence": 0.8},
            ],
            "errors": [], "fetched_at": 0.0, "ttl_seconds": 1800, "cache_age_seconds": 5,
        }

    monkeypatch.setattr(webapp.city_comparison, "get_comparison", fake_get_comparison)

    body = text_of(client.get("/compare"))
    assert "那覇" in body
    assert "秒前に取得" in body


@needs_models
def test_週間予報の画面が表示される(client):
    body = text_of(client.get("/week?city=東京"))

    assert "7日間" in body
    assert "あした" in body
    assert "再帰予測" in body


@needs_models
def test_週間予報は都市の指定が無ければトップへ戻す(client):
    assert client.get("/week").status_code == 302
