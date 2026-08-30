"""都市比較（city_comparison.py）のテスト。

外部への通信は行わない。ForecastService の代わりに、決まった結果を返す
差し替え用オブジェクトを使う。
"""

import pytest

import city_comparison
from outing_ml.forecasting import TomorrowForecast
from outing_ml.serve import Recommendation


def make_forecast(city, comfort_score, category="outdoor"):
    return TomorrowForecast(
        city=city, base_date="2026-08-30", target_date="2026-08-31",
        weather={"temperature": 22.0, "rain_probability": 10.0,
                "wind_speed": 2.0, "humidity": 50.0},
        recommendation=Recommendation(
            category=category, probabilities={category: 0.8},
            confidence=0.8, comfort_score=comfort_score,
            weather_type_name="過ごしやすい晴れの日",
        ),
    )


class FakeForecastService:
    """compare_tomorrow の代わりに、決まった結果を返す。呼ばれた回数を数える。"""

    def __init__(self, forecasts, errors=None):
        self.forecasts = forecasts
        self.errors = errors or []
        self.calls = 0

    def compare_tomorrow(self):
        self.calls += 1
        return {"results": self.forecasts, "errors": self.errors}


@pytest.fixture(autouse=True)
def clear_cache():
    city_comparison.clear_cache()
    yield
    city_comparison.clear_cache()


def test_日和度が高い順に並ぶ():
    service = FakeForecastService([
        make_forecast("東京", 60.0),
        make_forecast("那覇", 90.0),
        make_forecast("札幌", 30.0),
    ])

    data = city_comparison.get_comparison(service)

    assert [row["city"] for row in data["rankings"]] == ["那覇", "東京", "札幌"]
    assert data["cache_age_seconds"] == 0


def test_日和度が無いものは最後に回る():
    forecasts = [make_forecast("東京", 60.0)]
    forecasts[0].recommendation = None   # おすすめが無いケース

    service = FakeForecastService([make_forecast("那覇", 50.0), forecasts[0]])
    data = city_comparison.get_comparison(service)

    assert data["rankings"][-1]["city"] == "東京"
    assert data["rankings"][-1]["comfort_score"] is None


def test_失敗した都市はerrorsに入る():
    service = FakeForecastService(
        [make_forecast("東京", 60.0)],
        errors=[{"city": "青森", "message": "取得に失敗しました"}],
    )

    data = city_comparison.get_comparison(service)

    assert data["errors"] == [{"city": "青森", "message": "取得に失敗しました"}]
    assert len(data["rankings"]) == 1


def test_キャッシュが効く間は再取得しない():
    service = FakeForecastService([make_forecast("東京", 60.0)])

    first = city_comparison.get_comparison(service)
    second = city_comparison.get_comparison(service)

    assert service.calls == 1, "2回目はキャッシュを使うので呼ばれない"
    assert first["fetched_at"] == second["fetched_at"]
    assert second["cache_age_seconds"] >= 0


def test_強制更新はキャッシュを無視する():
    service = FakeForecastService([make_forecast("東京", 60.0)])

    city_comparison.get_comparison(service)
    city_comparison.get_comparison(service, force_refresh=True)

    assert service.calls == 2


def test_期限が切れたら取得し直す(monkeypatch):
    service = FakeForecastService([make_forecast("東京", 60.0)])

    city_comparison.get_comparison(service)
    monkeypatch.setattr(city_comparison, "CACHE_TTL_SECONDS", -1)   # 即座に期限切れにする
    city_comparison.get_comparison(service)

    assert service.calls == 2
