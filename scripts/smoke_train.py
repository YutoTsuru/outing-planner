"""学習が最後まで通ることを、少量のデータで確かめる（CI用）。

本番の学習は数分かかり、外部APIからのダウンロードも必要です。
CI では「コードが最後まで走るか」だけを、その場で作った小さなデータで確かめます。
成績の良し悪しは見ません（そこはテストと本番の学習の役目）。

実行方法:
    python scripts/smoke_train.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from outing_ml import features, labeling, metrics, monitor  # noqa: E402
from outing_ml.config import FEATURE_COLUMNS  # noqa: E402
from outing_ml.data import validate_frame  # noqa: E402
from outing_ml.registry import ModelBundle, Registry, load_bundle, save_bundle  # noqa: E402
from outing_ml.serve import OutingService  # noqa: E402


def make_dataset(days=200, cities=("東京", "大阪", "札幌")):
    """本物と同じ形の、小さな気象データを作る。"""
    rng = np.random.default_rng(0)
    calendar = pd.date_range("2023-01-01", periods=days, freq="D")

    rows = []
    for city in cities:
        for day in calendar:
            rows.append(
                {
                    "city": city,
                    "date": day.date(),
                    "temperature": round(float(rng.uniform(-5, 35)), 1),
                    "rain_probability": int(rng.integers(0, 11) * 10),
                    "wind_speed": round(float(rng.uniform(0.4, 12)), 1),
                    "humidity": int(rng.integers(30, 100)),
                }
            )
    return pd.DataFrame(rows)


def main():
    print("1. データを作って検証...")
    raw = make_dataset()
    validate_frame(raw).raise_if_failed()
    print(f"   {len(raw)} 行 OK")

    print("2. ラベル付け...")
    labeled = labeling.add_labels(raw)
    scored = labeling.add_comfort_scores(raw)
    print(f"   カテゴリ {labeling.label_distribution(labeled)} / "
          f"ベイズ限界 {labeling.bayes_accuracy(labeled):.3f}")
    assert scored["comfort_score"].between(0, 100).all()

    print("3. 時系列の特徴量...")
    supervised = features.build_supervised_frame(raw)
    features.assert_no_leakage(supervised)
    print(f"   {len(supervised)} 行 / 入力 {len(features.input_columns(supervised))} 列")

    print("4. 学習と評価...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X = labeled[FEATURE_COLUMNS]
    y = labeled["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    model = RandomForestClassifier(n_estimators=30, random_state=0).fit(X_train, y_train)
    predicted = model.predict(X_test)

    report = metrics.classification_metrics(
        y_test, predicted, model.predict_proba(X_test), list(model.classes_)
    )
    print(f"   正解率 {report['accuracy']['value']:.3f} "
          f"[{report['accuracy']['low']:.3f}〜{report['accuracy']['high']:.3f}]")

    print("5. 保存・読み込み・推論...")
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "model.pkl")
        entry = Registry(path=str(Path(folder) / "registry.json")).record(
            "smoke", path, "スモークテスト", {"accuracy": report["accuracy"]["value"]},
            {"sha256": "smoke"},
        )
        save_bundle(
            path,
            ModelBundle(
                estimator=model, feature_names=FEATURE_COLUMNS, model_name="smoke",
                version=entry["version"], task="スモークテスト",
                classes=list(model.classes_),
                metadata={"feature_ranges": {
                    column: {"min": float(raw[column].min()), "max": float(raw[column].max())}
                    for column in FEATURE_COLUMNS
                }},
            ),
        )
        service = OutingService(category=load_bundle(path))
        result = service.predict(22, 10, 2, 50)
        print(f"   予測: {result.category}（確信度 {result.confidence:.2f}）")
        assert result.category in ("indoor", "outdoor", "relax")

    print("6. ドリフト監視...")
    drift = monitor.drift_report(raw, raw)
    print(f"   同じデータどうし: {drift['overall']}")
    assert drift["overall"] == "OK"

    print("\nスモークテスト成功")


if __name__ == "__main__":
    main()
