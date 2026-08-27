"""データ検証のテスト。

「データが壊れていたら、静かに悪い予測を出すのではなく、その場で止まる」ことを守る。
"""

import pandas as pd
import pytest

from outing_ml.data import DataValidationError, validate_frame


def make_frame(**overrides):
    frame = pd.DataFrame(
        {
            "city": ["東京", "東京", "大阪"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
            "temperature": [10.0, 12.0, 14.0],
            "rain_probability": [0, 50, 100],
            "wind_speed": [2.0, 3.0, 4.0],
            "humidity": [60, 70, 80],
        }
    )
    for key, value in overrides.items():
        frame[key] = value
    return frame


def test_正しいデータは検証を通る():
    report = validate_frame(make_frame())
    assert report.ok
    assert report.rows == 3


def test_列が足りなければ落とす():
    frame = make_frame().drop(columns=["humidity"])
    report = validate_frame(frame)
    assert not report.ok
    assert any("humidity" in error for error in report.errors)


def test_欠損があれば落とす():
    frame = make_frame()
    frame.loc[0, "temperature"] = None
    report = validate_frame(frame)
    assert not report.ok
    assert any("欠損" in error for error in report.errors)


def test_数値でない値があれば落とす():
    report = validate_frame(make_frame(temperature=["あたたかい", "12", "14"]))
    assert not report.ok


@pytest.mark.parametrize(
    "column,value",
    [("humidity", [60, 70, 150]), ("rain_probability", [-10, 50, 100]),
     ("temperature", [10.0, 12.0, 99.0])],
)
def test_ありえない範囲の値は落とす(column, value):
    report = validate_frame(make_frame(**{column: value}))
    assert not report.ok


def test_同じ都市と日付が二重にあれば落とす():
    frame = make_frame()
    frame.loc[2, "city"] = "東京"
    frame.loc[2, "date"] = "2024-01-01"
    report = validate_frame(frame)
    assert not report.ok
    assert any("二重" in error or "同じ都市" in error for error in report.errors)


def test_日付がとびとびなら警告する():
    frame = make_frame()
    frame.loc[1, "date"] = "2024-01-10"
    report = validate_frame(frame)
    assert report.ok           # 使えないわけではない
    assert report.warnings     # だが気づけるようにする


def test_検証に失敗したら例外で止まる():
    report = validate_frame(make_frame().drop(columns=["city"]))
    with pytest.raises(DataValidationError):
        report.raise_if_failed()


def test_空のデータは落とす():
    report = validate_frame(make_frame().iloc[0:0])
    assert not report.ok
