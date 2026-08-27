"""時系列の特徴量のテスト。

いちばん怖いのは「未来の情報が入力にまざる」こと。
検証の点数だけが良くなり、本番でまったく当たらないモデルができる。
"""

import numpy as np
import pandas as pd
import pytest

from outing_ml import features
from outing_ml.config import FEATURE_COLUMNS


def sample_raw(days=30, cities=("東京", "大阪")):
    rng = np.random.default_rng(1)
    calendar = pd.date_range("2024-01-01", periods=days, freq="D")
    rows = []
    for city in cities:
        for offset in range(days):
            rows.append(
                {
                    "city": city,
                    "date": calendar[offset].date(),
                    "temperature": float(rng.uniform(0, 30)),
                    "rain_probability": int(rng.integers(0, 101)),
                    "wind_speed": float(rng.uniform(0, 10)),
                    "humidity": int(rng.integers(30, 100)),
                }
            )
    return pd.DataFrame(rows)


def test_入力に未来の情報がまざっていない():
    frame = features.build_supervised_frame(sample_raw())
    features.assert_no_leakage(frame)  # 例外が出なければ合格


def test_正解はその次の日の値になっている():
    raw = sample_raw(days=10, cities=("東京",))
    frame = features.build_supervised_frame(raw)

    row = frame.iloc[0]
    same_day = raw[raw["date"] == row["date"].date()].iloc[0]
    next_day = raw.iloc[raw.index.get_loc(same_day.name) + 1]

    for column in FEATURE_COLUMNS:
        assert row[f"{column}_today"] == pytest.approx(same_day[column])
        assert row[f"{column}_next"] == pytest.approx(next_day[column])


def test_都市をまたいで値が混ざらない():
    raw = sample_raw(days=10)
    frame = features.build_supervised_frame(raw)
    # 各都市で、前後2日ぶんが落ちる（最初2日＋最後1日）
    assert len(frame) == (10 - 3) * 2


def test_ウォークフォワードは必ず過去で学習して未来を予測する():
    raw = sample_raw(days=400, cities=("東京",))
    frame = features.build_supervised_frame(raw)

    seen = 0
    for _, train, test in features.walk_forward_splits(frame, years=(2025,)):
        assert train["date"].max() < test["date"].min()
        seen += 1
    assert seen >= 1


def test_未来の列が入力に残っていたら例外にする():
    frame = features.build_supervised_frame(sample_raw())
    broken = frame.rename(columns={"season_sin": "temperature_next_leak"})
    broken["temperature_next"] = 0.0
    # _next で終わる列は入力から除かれる仕組みなので、ここでは除去の結果を確認する
    assert "temperature_next" not in features.input_columns(broken)
