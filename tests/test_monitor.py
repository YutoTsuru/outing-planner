"""ドリフト監視のテスト。"""

import numpy as np
import pandas as pd

from outing_ml import monitor
from outing_ml.config import FEATURE_COLUMNS


def sample(n=2000, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "temperature": rng.normal(18 + shift, 8, n),
            "rain_probability": np.clip(rng.normal(26 + shift, 30, n), 0, 100),
            "wind_speed": np.clip(rng.normal(4, 2, n), 0, 20),
            "humidity": np.clip(rng.normal(67, 13, n), 0, 100),
        }
    )


def test_同じ分布ならPSIはほぼ0():
    values = sample()["temperature"].to_numpy()
    assert monitor.population_stability_index(values, values) < 0.01


def test_分布がずれるとPSIが大きくなる():
    reference = sample(seed=0)["temperature"].to_numpy()
    shifted = sample(shift=10, seed=1)["temperature"].to_numpy()
    assert monitor.population_stability_index(reference, shifted) > 0.25


def test_同じデータならレポートはOKになる():
    frame = sample()
    report = monitor.drift_report(frame, frame)
    assert report["overall"] == "OK"
    assert len(report["features"]) == len(FEATURE_COLUMNS)


def test_大きくずれたらALERTになる():
    report = monitor.drift_report(sample(seed=0), sample(shift=12, seed=1))
    assert report["overall"] == "ALERT"
    assert "学習し直し" in report["action"]


def test_件数が多いだけでは警告にしない():
    """件数が増えると、ごくわずかな差でも KS 検定の p値 は小さくなる。

    それで警告が出る監視は、いつも赤くなって誰も見なくなる。
    ずれの大きさ（効果量）で判断していることを確かめる。
    """
    report = monitor.drift_report(sample(n=20000, seed=0), sample(n=20000, seed=1))
    assert report["overall"] == "OK"
