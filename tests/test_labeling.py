"""ラベル生成のテスト。

ラベルの作り方が変わると、全モデルの意味が変わる。
「同じ入力なら必ず同じラベル」であることを機械で守る。
"""

import numpy as np
import pandas as pd

from outing_ml import labeling
from outing_ml.config import CATEGORIES


def sample_frame(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "temperature": rng.uniform(-10, 38, n),
            "rain_probability": rng.integers(0, 101, n),
            "wind_speed": rng.uniform(0, 15, n),
            "humidity": rng.integers(20, 101, n),
        }
    )


def test_確率は合計1になる():
    probabilities = labeling.label_probabilities(sample_frame())
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_同じ入力なら毎回同じラベルになる():
    frame = sample_frame()
    first = labeling.add_labels(frame)["label"].tolist()
    second = labeling.add_labels(frame)["label"].tolist()
    assert first == second


def test_ラベルは3種類のいずれか():
    labels = set(labeling.add_labels(sample_frame())["label"])
    assert labels <= set(CATEGORIES)


def test_ベイズ限界は3分の1以上1以下():
    limit = labeling.bayes_accuracy(labeling.add_labels(sample_frame()))
    assert 1 / 3 <= limit <= 1.0


def test_雨の日は屋内のおすすめ度がいちばん高い():
    frame = pd.DataFrame(
        [{"temperature": 20.0, "rain_probability": 100, "wind_speed": 3.0, "humidity": 80}]
    )
    probabilities = labeling.label_probabilities(frame)[0]
    assert CATEGORIES[int(probabilities.argmax())] == "indoor"


def test_過ごしやすい晴れの日は屋外のおすすめ度がいちばん高い():
    frame = pd.DataFrame(
        [{"temperature": 22.0, "rain_probability": 0, "wind_speed": 1.5, "humidity": 50}]
    )
    probabilities = labeling.label_probabilities(frame)[0]
    assert CATEGORIES[int(probabilities.argmax())] == "outdoor"


def test_日和度は0から100の範囲におさまる():
    scored = labeling.add_comfort_scores(sample_frame())
    assert scored["comfort_score"].between(0, 100).all()
    assert scored["true_score"].between(0, 100).all()


def test_快適な日ほど日和度が高い():
    frame = pd.DataFrame(
        [
            {"temperature": 22.0, "rain_probability": 0, "wind_speed": 1.0, "humidity": 50},
            {"temperature": 22.0, "rain_probability": 100, "wind_speed": 1.0, "humidity": 50},
        ]
    )
    scores = labeling.true_comfort_score(frame)
    assert scores[0] > scores[1]
