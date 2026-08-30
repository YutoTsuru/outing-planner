"""REST API のテスト。

外部への通信は行いません（翌日予報とプラン作成は差し替えて確かめます）。
確かめたいのは「入力が変なときに、静かに答えず、何が悪いか分かるエラーを返すこと」です。
"""

import os

import pytest

from outing_ml.config import CONFIG, FEATURE_COLUMNS
from outing_ml.serve import OutingService

MODELS_READY = os.path.exists(CONFIG.paths.category_model)
needs_models = pytest.mark.skipif(
    not MODELS_READY, reason="python train_all.py を先に実行してください"
)

GOOD_DAY = {"temperature": 22, "rain_probability": 10, "wind_speed": 2, "humidity": 50}
RAINY_DAY = {"temperature": 15, "rain_probability": 90, "wind_speed": 4, "humidity": 85}


class FakeForecast:
    """通信せずに、決まった予報を返す差し替え用。"""

    class _Bundle:
        version = "test"

    bundle = _Bundle()

    def __init__(self, outing):
        self.outing = outing

    def predict_tomorrow(self, city, days=12):
        from outing_ml.forecasting import TomorrowForecast

        weather = dict(GOOD_DAY)
        return TomorrowForecast(
            city=city,
            base_date="2026-08-27",
            target_date="2026-08-28",
            weather=weather,
            interval={"temperature": {"low": 20.0, "high": 24.0}},
            recommendation=self.outing.predict(**weather),
            model_version="test",
        )

    def predict_week(self, city, days_ahead=7, history_days=10):
        from outing_ml.forecasting import DailyForecast, WeeklyForecast

        days = [
            DailyForecast(
                day=day, date=f"2026-08-{27 + day:02d}", weather=dict(GOOD_DAY),
                interval={"temperature": {"low": 20.0 - day, "high": 24.0 + day}},
                recommendation=self.outing.predict(**GOOD_DAY),
            )
            for day in range(1, days_ahead + 1)
        ]
        return WeeklyForecast(city=city, base_date="2026-08-27", days=days,
                              model_version="test", notes=["再帰予測のテスト用"])


@pytest.fixture(autouse=True)
def clean_rate_limit():
    """レート制限のカウンタを、テストごとにきれいにする。

    グローバルな状態を持つモジュールなので、これが無いとテストの実行順で
    「前のテストが叩いた分」を引き継いでしまい、結果が不安定になる。
    """
    import rate_limit

    rate_limit.reset()
    yield
    rate_limit.reset()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """予測の記録を一時フォルダへ逃がしたテスト用クライアント。"""
    import api
    import prediction_log

    monkeypatch.setattr(prediction_log, "LOG_PATH", str(tmp_path / "predictions.jsonl"))

    outing = OutingService.load()
    app = api.create_api(outing=outing, forecast=FakeForecast(outing))
    app.config.update(TESTING=True)
    return app.test_client()


# ---------------------------------------------------------------
# 状態の確認
# ---------------------------------------------------------------

@needs_models
def test_起動確認で読み込めているモデルが分かる(client):
    body = client.get("/api/health").get_json()

    assert body["ok"]
    assert body["features"] == FEATURE_COLUMNS
    assert body["models"]["category"]


@needs_models
def test_学習の履歴が取れる(client):
    body = client.get("/api/models").get_json()

    assert body["ok"]
    assert body["models"], "1つも履歴が無い"
    first = body["models"][0]
    assert first["version"]
    assert first["data_sha256"], "どのデータで学習したかが残っている"


@needs_models
def test_都市と天気タイプの一覧が取れる(client):
    from outing_ml.config import CONFIG

    assert len(client.get("/api/cities").get_json()["cities"]) == len(CONFIG.data.cities)

    types = client.get("/api/weather-types").get_json()["types"]
    assert len(types) >= 3
    assert all(item["name"] for item in types)


# ---------------------------------------------------------------
# 予測
# ---------------------------------------------------------------

@needs_models
def test_予測が返る(client):
    body = client.post("/api/predict", json=GOOD_DAY).get_json()

    assert body["ok"]
    assert body["category"] in ("indoor", "outdoor", "relax")
    assert 0 <= body["confidence"] <= 1
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6
    assert 0 <= body["comfort_score"] <= 100
    assert body["weather_type_name"]


@needs_models
def test_雨の日は屋内がすすめられる(client):
    assert client.post("/api/predict", json=RAINY_DAY).get_json()["category"] == "indoor"


@needs_models
def test_項目が足りなければ何が足りないか教える(client):
    response = client.post("/api/predict", json={"temperature": 22})

    assert response.status_code == 400
    details = response.get_json()["error"]["details"]
    assert set(details["missing"]) == {"rain_probability", "wind_speed", "humidity"}
    assert "ranges" in details, "入力できる範囲も案内する"


@needs_models
def test_数値でない入力は400で理由を返す(client):
    response = client.post("/api/predict", json={**GOOD_DAY, "temperature": "あつい"})

    assert response.status_code == 400
    assert "数値" in response.get_json()["error"]["message"]


@needs_models
def test_JSONでない本文は400(client):
    response = client.post("/api/predict", data="ただの文字列",
                           content_type="text/plain")
    assert response.status_code == 400


@needs_models
def test_範囲外の値は丸めて警告を返す(client):
    body = client.post("/api/predict", json={**GOOD_DAY, "temperature": 99}).get_json()

    assert body["ok"], "予測は返す"
    assert any("範囲" in warning for warning in body["warnings"])


@needs_models
def test_学習範囲の外なら根拠が弱いと伝える(client):
    body = client.post("/api/predict", json={**GOOD_DAY, "temperature": 40}).get_json()
    assert any("学習データの範囲" in warning for warning in body["warnings"])


@needs_models
def test_まとめて予測できる(client):
    body = client.post("/api/predict/batch", json={"days": [GOOD_DAY, RAINY_DAY]}).get_json()

    assert body["count"] == 2
    assert [item["category"] for item in body["results"]] == ["outdoor", "indoor"]


@needs_models
def test_まとめて予測の上限を超えたら断る(client):
    import api

    response = client.post("/api/predict/batch",
                           json={"days": [GOOD_DAY] * (api.MAX_BATCH_SIZE + 1)})
    assert response.status_code == 400
    assert response.get_json()["error"]["details"]["received"] == api.MAX_BATCH_SIZE + 1


@needs_models
def test_まとめて予測は空配列を断る(client):
    assert client.post("/api/predict/batch", json={"days": []}).status_code == 400


@needs_models
def test_まとめて予測で何番目が悪いか分かる(client):
    response = client.post("/api/predict/batch",
                           json={"days": [GOOD_DAY, {"temperature": 22}]})

    assert response.status_code == 400
    assert "1 番目" in response.get_json()["error"]["message"]


# ---------------------------------------------------------------
# 翌日予報
# ---------------------------------------------------------------

@needs_models
def test_あしたの予報とおすすめが返る(client):
    body = client.get("/api/forecast?city=東京").get_json()

    assert body["ok"]
    assert body["target_date"] == "2026-08-28"
    assert set(body["weather"]) == set(FEATURE_COLUMNS)
    assert body["recommendation"]["category"]


@needs_models
def test_都市の指定が無ければ使える都市を案内する(client):
    response = client.get("/api/forecast")

    assert response.status_code == 400
    assert "東京" in response.get_json()["error"]["details"]["cities"]


@needs_models
def test_日数の指定が範囲外なら断る(client):
    assert client.get("/api/forecast?city=東京&days=999").status_code == 400
    assert client.get("/api/forecast?city=東京&days=abc").status_code == 400


# ---------------------------------------------------------------
# プラン
# ---------------------------------------------------------------

@needs_models
def test_プランのカテゴリが正しくなければ断る(client):
    response = client.post("/api/plan", json={"category": "宇宙旅行", "area": "京都市"})

    assert response.status_code == 400
    assert "outdoor" in response.get_json()["error"]["details"]["allowed"]


@needs_models
def test_プランは場所の指定が要る(client):
    assert client.post("/api/plan", json={"category": "outdoor"}).status_code == 400


@needs_models
def test_プランが作れる(client, monkeypatch):
    # 外部への通信をせずに、組み立てまで通ることだけを確かめる
    import api

    monkeypatch.setattr(api.planner, "build_plan",
                        lambda *args, **kwargs: "### テスト用のプラン")

    body = client.post("/api/plan", json={
        "category": "outdoor", "latitude": 35.0, "longitude": 135.0,
        "area": "テスト市", "start_time": "10:00", "end_time": "15:00",
    }).get_json()

    assert body["ok"]
    assert body["plan_markdown"].startswith("### ")


@needs_models
def test_プランの範囲指定が広すぎたら断る(client):
    response = client.post("/api/plan", json={
        "category": "outdoor", "latitude": 35.0, "longitude": 135.0, "radius_km": 100,
    })
    assert response.status_code == 400


# ---------------------------------------------------------------
# そのほか
# ---------------------------------------------------------------

@needs_models
def test_無いURLとメソッド違いはJSONで返す(client):
    not_found = client.get("/api/nothing-here")
    assert not_found.status_code == 404
    assert not_found.get_json()["ok"] is False

    wrong_method = client.get("/api/predict")
    assert wrong_method.status_code == 405
    assert wrong_method.get_json()["ok"] is False


@needs_models
def test_予測を記録している(client, tmp_path):
    import json

    import prediction_log

    client.post("/api/predict", json=GOOD_DAY)

    assert os.path.exists(prediction_log.LOG_PATH), "使われた天気を残しておく"
    with open(prediction_log.LOG_PATH, encoding="utf-8") as file:
        line = json.loads(file.readline())
    assert line["kind"] == "predict"
    assert line["input"] == GOOD_DAY


@needs_models
def test_記録を履歴として取り出せる(client):
    client.post("/api/predict", json=GOOD_DAY)
    client.post("/api/predict", json=RAINY_DAY)

    body = client.get("/api/history").get_json()

    assert body["ok"]
    assert body["logging_enabled"] is True
    assert body["summary"]["total"] >= 2
    assert body["entries"][0]["kind"] == "predict"
    assert body["entries"][0]["category"] == "indoor", "新しい順に返る"


@needs_models
def test_履歴の件数を絞れる(client):
    for _ in range(3):
        client.post("/api/predict", json=GOOD_DAY)

    assert len(client.get("/api/history?limit=2").get_json()["entries"]) == 2


@needs_models
def test_履歴の件数指定が範囲外なら断る(client):
    assert client.get("/api/history?limit=0").status_code == 400
    assert client.get("/api/history?limit=abc").status_code == 400


@needs_models
def test_記録がなくても履歴は返る(client):
    body = client.get("/api/history").get_json()

    assert body["ok"]
    assert body["summary"]["total"] == 0
    assert body["entries"] == []


@needs_models
def test_ドリフト監視のレポートが返る(client):
    body = client.get("/api/monitor").get_json()

    assert body["ok"]
    assert body["overall"] in ("OK", "WATCH", "ALERT")
    assert len(body["features"]) == 4
    assert "current_source" in body


@needs_models
def test_都市比較のランキングが返る(client, monkeypatch):
    import api

    def fake_get_comparison(service, force_refresh=False):
        return {
            "rankings": [
                {"city": "那覇", "target_date": "2026-08-31",
                 "weather": {"temperature": 28.0, "rain_probability": 5.0,
                            "wind_speed": 3.0, "humidity": 60.0},
                 "category": "outdoor", "comfort_score": 90.0,
                 "weather_type_name": "過ごしやすい晴れの日", "confidence": 0.8},
            ],
            "errors": [], "fetched_at": 0.0, "ttl_seconds": 1800, "cache_age_seconds": 0,
        }

    monkeypatch.setattr(api.city_comparison, "get_comparison", fake_get_comparison)

    body = client.get("/api/compare").get_json()
    assert body["ok"]
    assert body["rankings"][0]["city"] == "那覇"


@needs_models
def test_都市比較で強制更新を渡せる(client, monkeypatch):
    import api

    captured = {}

    def fake_get_comparison(service, force_refresh=False):
        captured["force_refresh"] = force_refresh
        return {"rankings": [], "errors": [], "fetched_at": 0.0,
               "ttl_seconds": 1800, "cache_age_seconds": 0}

    monkeypatch.setattr(api.city_comparison, "get_comparison", fake_get_comparison)

    client.get("/api/compare?refresh=1")
    assert captured["force_refresh"] is True


@needs_models
def test_週間予報が返る(client):
    body = client.get("/api/week?city=東京").get_json()

    assert body["ok"]
    assert len(body["days"]) == 7
    assert body["days"][0]["day"] == 1
    assert body["notes"]


@needs_models
def test_週間予報の日数指定を絞れる(client):
    body = client.get("/api/week?city=東京&days=3").get_json()
    assert len(body["days"]) == 3


@needs_models
def test_週間予報の日数指定が範囲外なら断る(client):
    assert client.get("/api/week?city=東京&days=999").status_code == 400
    assert client.get("/api/week?city=東京&days=0").status_code == 400


@needs_models
def test_週間予報は都市の指定が要る(client):
    response = client.get("/api/week")
    assert response.status_code == 400
    assert "東京" in response.get_json()["error"]["details"]["cities"]


@needs_models
def test_レート制限を超えると429になる(client):
    for _ in range(60):
        assert client.post("/api/predict", json=GOOD_DAY).status_code == 200

    response = client.post("/api/predict", json=GOOD_DAY)
    assert response.status_code == 429
    assert response.headers["Retry-After"]
    assert response.get_json()["error"]["details"]["limit_per_minute"] == 60


@needs_models
def test_成功したレスポンスにも残り回数が入る(client):
    response = client.post("/api/predict", json=GOOD_DAY)

    assert response.headers["X-RateLimit-Limit"] == "60"
    assert response.headers["X-RateLimit-Remaining"] == "59"


@needs_models
def test_healthとopenapiはレート制限の対象外(client):
    for _ in range(65):
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/openapi.json").status_code == 200


@needs_models
def test_エンドポイントごとに別々に数える(client):
    for _ in range(60):
        client.post("/api/predict", json=GOOD_DAY)

    # predict は上限に達しているが、weather-types は別カウントなので通る
    assert client.get("/api/weather-types").status_code == 200


@needs_models
def test_都市比較の強制更新には別枠の厳しい上限がある(client, monkeypatch):
    import api

    monkeypatch.setattr(api.city_comparison, "get_comparison",
                        lambda service, force_refresh=False: {
                            "rankings": [], "errors": [], "fetched_at": 0.0,
                            "ttl_seconds": 1800, "cache_age_seconds": 0,
                        })

    for _ in range(api._COMPARE_REFRESH_LIMIT_PER_MINUTE):
        assert client.get("/api/compare?refresh=1").status_code == 200

    response = client.get("/api/compare?refresh=1")
    assert response.status_code == 429
