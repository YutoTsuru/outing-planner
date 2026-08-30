"""予測の記録のテスト。"""

import json
import os

import prediction_log
from outing_ml.config import CATEGORIES


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")


def test_書いたものを新しい順で読み出せる(tmp_path):
    path = str(tmp_path / "log.jsonl")
    for index in range(3):
        prediction_log.append("predict", {"n": index}, path=path)

    entries = prediction_log.read_entries(path=path)
    assert [entry["n"] for entry in entries] == [2, 1, 0]
    assert all("at" in entry for entry in entries)


def test_件数を絞れる(tmp_path):
    path = str(tmp_path / "log.jsonl")
    for index in range(10):
        prediction_log.append("predict", {"n": index}, path=path)

    assert len(prediction_log.read_entries(limit=3, path=path)) == 3


def test_種類で絞れる(tmp_path):
    path = str(tmp_path / "log.jsonl")
    prediction_log.append("predict", {"n": 1}, path=path)
    prediction_log.append("forecast", {"n": 2}, path=path)

    entries = prediction_log.read_entries(kinds={"forecast"}, path=path)
    assert len(entries) == 1
    assert entries[0]["kind"] == "forecast"


def test_記録がなければ空を返す(tmp_path):
    assert prediction_log.read_entries(path=str(tmp_path / "ない.jsonl")) == []


def test_壊れた行は飛ばして読む(tmp_path):
    """追記中のファイルを読むので、最後の行が途中で切れていることがある。"""
    path = str(tmp_path / "log.jsonl")
    write_lines(path, [
        json.dumps({"at": "2026-08-30T00:00:00+00:00", "kind": "predict", "n": 1}),
        '{"at": "2026-08-30T00:00:01+00:00", "kind": "pre',   # 途中で切れた行
        json.dumps({"at": "2026-08-30T00:00:02+00:00", "kind": "predict", "n": 2}),
    ])

    entries = prediction_log.read_entries(path=path)
    assert [entry["n"] for entry in entries] == [2, 1]


def test_記録がオフなら書かない(tmp_path, monkeypatch):
    monkeypatch.setenv(prediction_log.ENV_FLAG, "0")
    path = str(tmp_path / "log.jsonl")

    prediction_log.append("predict", {"n": 1}, path=path)
    assert not os.path.exists(path)
    assert prediction_log.enabled() is False


def test_書き込みに失敗しても例外にしない(tmp_path):
    """記録は本筋ではないので、失敗しても予測は返す。"""
    path = str(tmp_path / "log.jsonl")
    os.makedirs(path)          # 同名のディレクトリを作って書き込みを失敗させる

    prediction_log.append("predict", {"n": 1}, path=path)   # 例外が出なければ合格


def test_傾向をまとめられる():
    entries = [
        {"at": "2026-08-30T10:00:00+00:00", "kind": "predict", "category": "outdoor",
         "confidence": 0.9,
         "input": {"temperature": 20.0, "rain_probability": 0.0,
                   "wind_speed": 2.0, "humidity": 50.0}},
        {"at": "2026-08-30T09:00:00+00:00", "kind": "predict", "category": "outdoor",
         "confidence": 0.7,
         "input": {"temperature": 24.0, "rain_probability": 20.0,
                   "wind_speed": 4.0, "humidity": 70.0}},
        {"at": "2026-08-30T08:00:00+00:00", "kind": "web_predict", "category": "indoor",
         "confidence": 0.8,
         "input": {"temperature": 16.0, "rain_probability": 100.0,
                   "wind_speed": 6.0, "humidity": 90.0}},
    ]

    summary = prediction_log.summarize(entries)

    assert summary["total"] == 3
    assert summary["by_kind"] == {"predict": 2, "web_predict": 1}
    assert summary["averages"]["temperature"] == 20.0
    assert summary["average_confidence"] == 0.8
    assert summary["period"]["from"].endswith("08:00:00+00:00")
    assert summary["period"]["to"].endswith("10:00:00+00:00")

    shares = {row["category"]: row for row in summary["by_category"]}
    assert set(shares) == set(CATEGORIES), "出ていないカテゴリも0件として並べる"
    assert shares["outdoor"]["count"] == 2
    assert shares["relax"]["count"] == 0


def test_記録が空でもまとめは落ちない():
    summary = prediction_log.summarize([])

    assert summary["total"] == 0
    assert summary["period"] is None
    assert summary["average_confidence"] is None
