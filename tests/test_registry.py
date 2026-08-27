"""成果物の保存と履歴のテスト。"""

import json
import os

from sklearn.dummy import DummyClassifier

from outing_ml.registry import ModelBundle, Registry, load_bundle, save_bundle


def make_bundle():
    estimator = DummyClassifier(strategy="prior").fit([[1, 2, 3, 4]], ["indoor"])
    return ModelBundle(
        estimator=estimator,
        feature_names=["temperature", "rain_probability", "wind_speed", "humidity"],
        model_name="test-model",
        version="v1",
        task="テスト用",
        classes=["indoor"],
        metadata={"data": {"sha256": "abc"}},
    )


def test_保存して読み込むと同じ内容になる(tmp_path):
    path = os.path.join(tmp_path, "model.pkl")
    save_bundle(path, make_bundle())

    loaded = load_bundle(path)
    assert loaded.model_name == "test-model"
    assert loaded.feature_names[0] == "temperature"
    assert loaded.metadata["data"]["sha256"] == "abc"


def test_特徴量の順番が違えば例外になる():
    bundle = make_bundle()
    bundle.check_features(bundle.feature_names)  # 同じなら通る

    wrong = list(reversed(bundle.feature_names))
    try:
        bundle.check_features(wrong)
    except ValueError as error:
        assert "学習時" in str(error)
    else:
        raise AssertionError("順番違いが素通りしました")


def test_履歴は積み上がり版が増える(tmp_path):
    registry = Registry(path=os.path.join(tmp_path, "registry.json"))

    first = registry.record("m", "a.pkl", "task", {"acc": 0.8}, {"sha256": "x"})
    second = registry.record("m", "a.pkl", "task", {"acc": 0.9}, {"sha256": "y"})

    assert first["version"] == "v1"
    assert second["version"] == "v2"
    assert registry.latest("m")["version"] == "v2"
    assert len(registry.history("m")) == 2


def test_履歴にはデータの指紋と環境が残る(tmp_path):
    registry = Registry(path=os.path.join(tmp_path, "registry.json"))
    entry = registry.record("m", "a.pkl", "task", {"acc": 0.8}, {"sha256": "x"})

    assert entry["data"]["sha256"] == "x"
    assert "scikit_learn" in entry["environment"]
    assert "created_at" in entry

    with open(os.path.join(tmp_path, "registry.json"), encoding="utf-8") as file:
        assert json.load(file)["entries"][0]["model_name"] == "m"


def test_無いファイルを読むと分かりやすい例外になる(tmp_path):
    try:
        load_bundle(os.path.join(tmp_path, "no-such-model.pkl"))
    except FileNotFoundError as error:
        assert "train_all" in str(error)
    else:
        raise AssertionError("見つからないのに例外になりませんでした")
