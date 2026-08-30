"""プランの共有リンク（shared_plans.py）のテスト。"""


import pytest

import shared_plans


@pytest.fixture(autouse=True)
def temp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_plans, "STORE_PATH", str(tmp_path / "shared_plans.jsonl"))


def test_保存すると短いIDが返る():
    plan_id = shared_plans.save("outdoor", "京都府京都市", "### プラン本文")

    assert isinstance(plan_id, str)
    assert len(plan_id) == 12


def test_保存したものをIDで取り出せる():
    plan_id = shared_plans.save("outdoor", "京都府京都市", "### プラン本文")

    entry = shared_plans.find(plan_id)

    assert entry["id"] == plan_id
    assert entry["category"] == "outdoor"
    assert entry["area"] == "京都府京都市"
    assert entry["plan_markdown"] == "### プラン本文"
    assert "created_at" in entry


def test_存在しないIDは例外になる():
    with pytest.raises(shared_plans.PlanNotFoundError):
        shared_plans.find("no-such-id")


def test_ファイルが無くても例外になる():
    with pytest.raises(shared_plans.PlanNotFoundError):
        shared_plans.find("anything")


def test_複数件から正しいものを見つけられる():
    first = shared_plans.save("outdoor", "京都市", "1件目")
    second = shared_plans.save("indoor", "大阪市", "2件目")

    assert shared_plans.find(first)["plan_markdown"] == "1件目"
    assert shared_plans.find(second)["plan_markdown"] == "2件目"


def test_IDが違う人のプランと混ざらない():
    for index in range(5):
        shared_plans.save("outdoor", f"都市{index}", f"プラン{index}")

    entry = shared_plans.find(shared_plans.save("relax", "最後の都市", "最後のプラン"))
    assert entry["plan_markdown"] == "最後のプラン"


def test_壊れた行があっても読み飛ばす(tmp_path):
    plan_id = shared_plans.save("outdoor", "京都市", "正しいプラン")

    with open(shared_plans.STORE_PATH, "a", encoding="utf-8") as file:
        file.write("これはJSONではない行\n")

    assert shared_plans.find(plan_id)["plan_markdown"] == "正しいプラン"


