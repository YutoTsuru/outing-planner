"""評価まわりのテスト。"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from outing_ml import metrics


def test_信頼区間は点推定をはさむ():
    rng = np.random.default_rng(0)
    y_true = rng.choice(["a", "b", "c"], 500)
    y_pred = y_true.copy()
    mask = rng.random(500) < 0.2
    y_pred[mask] = rng.choice(["a", "b", "c"], mask.sum())

    interval = metrics.bootstrap_interval(accuracy_score, y_true, y_pred, n_samples=200)
    assert interval["low"] <= interval["value"] <= interval["high"]


def test_完全に較正された確率のECEは小さい():
    rng = np.random.default_rng(0)
    classes = ["a", "b"]
    confidence = rng.uniform(0.5, 1.0, 5000)
    proba = np.column_stack([confidence, 1 - confidence])
    y_true = np.where(rng.random(5000) < confidence, "a", "b")

    result = metrics.expected_calibration_error(y_true, proba, classes, bins=10)
    assert result["ece"] < 0.05


def test_自信過剰な確率のECEは大きい():
    classes = ["a", "b"]
    proba = np.tile([0.99, 0.01], (1000, 1))
    y_true = np.array(["a"] * 500 + ["b"] * 500)

    result = metrics.expected_calibration_error(y_true, proba, classes, bins=10)
    assert result["ece"] > 0.3


def test_同じ予測どうしのマクネマー検定は有意にならない():
    y_true = np.array(["a", "b"] * 50)
    result = metrics.mcnemar_test(y_true, y_true, y_true)
    assert not result["significant"]


def test_片方が明らかに良ければマクネマー検定は有意になる():
    y_true = np.array(["a"] * 100)
    good = np.array(["a"] * 100)
    bad = np.array(["b"] * 100)
    result = metrics.mcnemar_test(y_true, good, bad)
    assert result["significant"]
    assert result["only_a"] == 100


def test_スライス評価はグループごとに成績を返す():
    frame = pd.DataFrame({"city": ["東京"] * 50 + ["大阪"] * 50})
    y_true = np.array(["a"] * 100)
    y_pred = np.array(["a"] * 50 + ["b"] * 50)

    rows = metrics.slice_report(frame, y_true, y_pred, "city")
    assert len(rows) == 2
    assert rows[0]["accuracy"] == 0.0      # 悪いほうが先頭
    assert rows[-1]["accuracy"] == 1.0


def test_予測区間のカバレッジ():
    assert metrics.interval_coverage([1, 2, 3], [0, 0, 0], [2, 2, 2]) == 2 / 3


def test_ピンボール損失は中央値で対称になる():
    assert metrics.pinball_loss([2.0], [1.0], 0.5) == metrics.pinball_loss([0.0], [1.0], 0.5)


def test_季節の割り当て():
    seasons = metrics.season_of(["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15"])
    assert list(seasons) == ["冬", "春", "夏", "秋"]
