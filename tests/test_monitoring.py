"""アプリのドリフト監視（monitoring.py）のテスト。

外部のモデルファイルやCSVには触れず、prediction_log と outing_ml.data だけ差し替える。
"""

import os

import pandas as pd
import pytest

import monitoring
import prediction_log
from outing_ml.config import CONFIG, FEATURE_COLUMNS

MODELS_READY = os.path.exists(CONFIG.paths.category_model)
needs_models = pytest.mark.skipif(
    not MODELS_READY, reason="python train_all.py を先に実行してください"
)


def sample_frame(n, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "temperature": rng.normal(18, 8, n),
            "rain_probability": rng.uniform(0, 100, n),
            "wind_speed": rng.uniform(0, 10, n),
            "humidity": rng.uniform(30, 90, n),
        }
    )


def test_記録が少なければholdoutにフォールバックする(monkeypatch, tmp_path):
    log_path = str(tmp_path / "predictions.jsonl")
    monkeypatch.setattr(prediction_log, "LOG_PATH", log_path)

    frame, source, note = monitoring._choose_current_source()

    assert source == "holdout_data"
    assert "足りない" in note


def test_十分な記録があればそれを使う(monkeypatch, tmp_path):
    log_path = str(tmp_path / "predictions.jsonl")
    monkeypatch.setattr(prediction_log, "LOG_PATH", log_path)
    monkeypatch.setattr(monitoring, "MIN_LIVE_ROWS", 5)

    for offset in range(10):
        prediction_log.append(
            "predict",
            {"input": {"temperature": 20.0 + offset, "rain_probability": 10.0,
                      "wind_speed": 2.0, "humidity": 50.0}},
        )

    frame, source, note = monitoring._choose_current_source()

    assert source == "live_predictions"
    assert len(frame) == 10


def test_holdoutも記録も無ければ例外にする(monkeypatch, tmp_path):
    monkeypatch.setattr(prediction_log, "LOG_PATH", str(tmp_path / "predictions.jsonl"))
    monkeypatch.setattr(os.path, "exists", lambda path: False)

    with pytest.raises(monitoring.MonitorUnavailableError):
        monitoring._choose_current_source()


@needs_models
def test_全体のレポートが組み立てられる(monkeypatch, tmp_path):
    monkeypatch.setattr(prediction_log, "LOG_PATH", str(tmp_path / "predictions.jsonl"))

    report = monitoring.monitor_report()

    assert report["overall"] in ("OK", "WATCH", "ALERT")
    assert len(report["features"]) == len(FEATURE_COLUMNS)
    assert report["current_source"] in ("live_predictions", "holdout_data")
    assert "action" in report


@needs_models
def test_大きくずれたレポートはALERTになる(monkeypatch, tmp_path):
    monkeypatch.setattr(prediction_log, "LOG_PATH", str(tmp_path / "predictions.jsonl"))
    monkeypatch.setattr(monitoring, "MIN_LIVE_ROWS", 5)

    # 学習データとまったく違う天気（真夏の東南アジアのような値）を大量に記録する
    for _ in range(50):
        prediction_log.append(
            "predict",
            {"input": {"temperature": 39.0, "rain_probability": 95.0,
                      "wind_speed": 15.0, "humidity": 95.0}},
        )

    report = monitoring.monitor_report()
    assert report["overall"] == "ALERT"
    assert "学習し直し" in report["action"]
