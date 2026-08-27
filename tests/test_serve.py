"""推論API（サービング）のテスト。

「変な値を渡したら、静かに答えるのではなく、警告か例外にする」ことを守る。
"""

import os

import pandas as pd
import pytest

from outing_ml.config import CONFIG, FEATURE_COLUMNS
from outing_ml.serve import InvalidInputError, OutingService, validate_weather

MODELS_READY = os.path.exists(CONFIG.paths.category_model)
needs_models = pytest.mark.skipif(
    not MODELS_READY, reason="python train_all.py を先に実行してください"
)


def test_正しい入力はそのまま通る():
    values, warnings = validate_weather(
        {"temperature": 22, "rain_probability": 10, "wind_speed": 2, "humidity": 50}
    )
    assert values["temperature"] == 22.0
    assert warnings == []


def test_範囲の外は丸めて警告する():
    values, warnings = validate_weather(
        {"temperature": 99, "rain_probability": -5, "wind_speed": 2, "humidity": 50}
    )
    assert values["temperature"] == 40.0
    assert values["rain_probability"] == 0.0
    assert len(warnings) == 2


def test_数値でない入力は例外にする():
    with pytest.raises(InvalidInputError):
        validate_weather(
            {"temperature": "あつい", "rain_probability": 10, "wind_speed": 2, "humidity": 50}
        )


def test_足りない項目は例外にする():
    with pytest.raises(InvalidInputError):
        validate_weather({"temperature": 22, "rain_probability": 10, "wind_speed": 2})


def test_NaNは例外にする():
    with pytest.raises(InvalidInputError):
        validate_weather(
            {"temperature": float("nan"), "rain_probability": 10,
             "wind_speed": 2, "humidity": 50}
        )


@needs_models
def test_予測が返る():
    service = OutingService.load()
    result = service.predict(22, 10, 2, 50)

    assert result.category in ("indoor", "outdoor", "relax")
    assert 0.0 <= result.confidence <= 1.0
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6


@needs_models
def test_過ごしやすい日は屋外がすすめられる():
    service = OutingService.load()
    assert service.predict(22, 0, 1.5, 50).category == "outdoor"


@needs_models
def test_雨の日は屋内がすすめられる():
    service = OutingService.load()
    assert service.predict(18, 90, 3, 80).category == "indoor"


@needs_models
def test_日和度と天気タイプも返る():
    service = OutingService.load()
    result = service.predict(22, 10, 2, 50)

    if service.comfort is not None:
        assert 0 <= result.comfort_score <= 100
    if service.weather_type is not None:
        assert isinstance(result.weather_type, int)
        assert result.weather_type_name


@needs_models
def test_まとめて予測できる():
    service = OutingService.load()
    frame = pd.DataFrame(
        [[22, 10, 2, 50], [15, 90, 4, 85]], columns=FEATURE_COLUMNS
    )
    result = service.predict_batch(frame)

    assert len(result) == 2
    assert "category" in result.columns
    assert result["confidence"].between(0, 1).all()


@needs_models
def test_起動確認の情報が取れる():
    health = OutingService.load().health()
    assert health["ok"]
    assert health["features"] == FEATURE_COLUMNS
    assert health["models"]["category"]


@needs_models
def test_学習範囲の外なら警告がつく():
    service = OutingService.load()
    result = service.predict(40, 10, 2, 50)   # 学習データの最高気温は 36.6℃
    assert any("学習データの範囲" in warning for warning in result.warnings)
