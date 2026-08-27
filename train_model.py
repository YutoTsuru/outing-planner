"""
お出かけプランナー：カテゴリ予測モデルの学習スクリプト

実測の気象データ（気温・降水確率・風速・湿度）から、
おすすめのお出かけカテゴリ（outdoor / indoor / relax）を予測するモデルを作ります。

このスクリプトは、単に学習して保存するだけではありません。
本番で使うモデルに必要な確認を、順番にすべて行います。

    1. データを検証する（列・型・範囲・重複・欠測）
    2. 正解ラベルを作る（おすすめ度モデル＋抽選）
    3. 候補5つを交差検証で比べ、いちばん良いものを選ぶ
    4. 確率の質（較正）を確かめ、必要なら較正しなおす
    5. 評価用データで、信頼区間つきの成績を出す
    6. 都市別・季節別の内訳を見て、苦手なところを探す
    7. 学習後に一度も見ていない「未来のデータ」で最終確認する
    8. 特徴量・データの指紋・コミットを付けて保存し、履歴に記録する

実行方法:
    python train_model.py
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from outing_ml import data as data_module
from outing_ml import labeling, metrics
from outing_ml.config import CATEGORIES, CONFIG, FEATURE_COLUMNS
from outing_ml.registry import ModelBundle, Registry, save_bundle

MODEL_NAME = "category-classifier"
SEED = CONFIG.train.random_seed

# ほかの学習スクリプトから参照される値（互換のために残している）
DATA_PATH = CONFIG.paths.dataset
MODEL_DIR = CONFIG.paths.model_dir
MODEL_PATH = CONFIG.paths.category_model
CARD_PATH = CONFIG.paths.category_card

discomfort_index = labeling.discomfort_index
add_labels = labeling.add_labels
bayes_accuracy = labeling.bayes_accuracy


def load_dataset(path: str = None):
    """気象データを読み込む（無ければダウンロードする）。"""
    path = path or DATA_PATH

    if not os.path.exists(path):
        print(f"   {path} が無いので、先にダウンロードします")
        import fetch_weather

        frame = fetch_weather.download_all()
        os.makedirs(CONFIG.paths.data_dir, exist_ok=True)
        frame.to_csv(path, index=False)

    return data_module.load_dataset(path)


# ---------------------------------------------------------------
# モデルの候補
# ---------------------------------------------------------------

def build_candidates():
    """比べるモデルの一覧を作る。

    まず「いちばん多いカテゴリを答えるだけ」のものさし（ベースライン）を置き、
    そこからどれだけ良くなったかで、機械学習の効果を確かめます。
    """
    return {
        "ベースライン（最多クラス）": DummyClassifier(strategy="prior", random_state=SEED),
        "ロジスティック回帰": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
            ]
        ),
        "決定木（深さ6）": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, random_state=SEED
        ),
        "ランダムフォレスト": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED
        ),
        "勾配ブースティング": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=SEED,
        ),
    }


def compare_candidates(X_train, y_train):
    """交差検証で候補を比べて、成績の表（DataFrame）を返す。"""
    cv = StratifiedKFold(n_splits=CONFIG.train.cv_splits, shuffle=True, random_state=SEED)
    rows = []

    for name, model in build_candidates().items():
        scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=["accuracy", "f1_macro", "neg_log_loss"],
        )
        rows.append(
            {
                "モデル": name,
                "正解率": scores["test_accuracy"].mean(),
                "マクロF1": scores["test_f1_macro"].mean(),
                "マクロF1の標準偏差": scores["test_f1_macro"].std(),
                "対数損失": -scores["test_neg_log_loss"].mean(),
            }
        )

    return pd.DataFrame(rows).sort_values("マクロF1", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------
# 較正（確率の質）
# ---------------------------------------------------------------

def calibrate_if_better(base_model, X_train, y_train, X_test, y_test):
    """確率が正直かどうかを確かめ、較正したほうが良ければ差し替える。

    アプリは「おすすめ度 85%」のように確率を見せることがあります。
    そのとき、85% と言った予測が本当に85%くらい当たっていないと、
    数字そのものが嘘になってしまいます。
    ここでは較正前後を比べ、対数損失が小さいほうを採用します。
    """
    from sklearn.base import clone

    plain = clone(base_model).fit(X_train, y_train)
    plain_proba = plain.predict_proba(X_test)
    plain_classes = list(plain.classes_)
    plain_ece = metrics.expected_calibration_error(y_test, plain_proba, plain_classes)
    plain_loss = float(log_loss(y_test, plain_proba, labels=plain_classes))

    calibrated = CalibratedClassifierCV(
        clone(base_model), method="isotonic", cv=CONFIG.train.cv_splits
    ).fit(X_train, y_train)
    calibrated_proba = calibrated.predict_proba(X_test)
    calibrated_classes = list(calibrated.classes_)
    calibrated_ece = metrics.expected_calibration_error(
        y_test, calibrated_proba, calibrated_classes
    )
    calibrated_loss = float(
        log_loss(y_test, calibrated_proba, labels=calibrated_classes)
    )

    use_calibrated = calibrated_loss < plain_loss

    report = {
        "before": {"ece": plain_ece["ece"], "log_loss": plain_loss,
                   "bins": plain_ece["bins"]},
        "after": {"ece": calibrated_ece["ece"], "log_loss": calibrated_loss,
                  "bins": calibrated_ece["bins"]},
        "calibrated": use_calibrated,
        "method": "isotonic" if use_calibrated else "none",
    }

    return (calibrated if use_calibrated else plain), report


# ---------------------------------------------------------------
# 評価
# ---------------------------------------------------------------

def evaluate(model, X_test, y_test, frame_test, baseline_pred, runner_up_pred):
    """評価用データでの成績を、信頼区間・内訳・確率の質までまとめて返す。"""
    predicted = model.predict(X_test)
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)

    result = metrics.classification_metrics(y_test, predicted, proba, classes)
    result["per_class"] = classification_report(
        y_test, predicted, labels=CATEGORIES, output_dict=True, zero_division=0
    )
    result["calibration"] = metrics.expected_calibration_error(y_test, proba, classes)
    result["vs_baseline"] = metrics.mcnemar_test(y_test, predicted, baseline_pred)
    result["vs_runner_up"] = metrics.mcnemar_test(y_test, predicted, runner_up_pred)

    seasons = metrics.season_of(frame_test["date"])
    result["slices"] = {
        "city": metrics.slice_report(frame_test, y_test, predicted, "city"),
        "season": metrics.slice_report(
            frame_test.assign(season=seasons), y_test, predicted, "season"
        ),
    }

    return result


def evaluate_holdout(model, path: str = None):
    """学習中に一度も見ていない「未来のデータ」で最終確認する。"""
    path = path or CONFIG.paths.holdout
    if not os.path.exists(path):
        return None

    frame = data_module.load_dataset(path)
    labeled = labeling.add_labels(frame)
    X = labeled[FEATURE_COLUMNS]
    y = labeled["label"]

    predicted = model.predict(X)
    proba = model.predict_proba(X)
    classes = list(model.classes_)

    result = metrics.classification_metrics(y, predicted, proba, classes)
    result["bayes_accuracy"] = labeling.bayes_accuracy(labeled)
    result["rows"] = int(len(labeled))
    result["date_from"] = str(frame["date"].min())
    result["date_to"] = str(frame["date"].max())
    result["calibration"] = metrics.expected_calibration_error(y, proba, classes)
    return result


def feature_importances(model, X_test, y_test):
    """どの入力が予測に効いているかを調べる（並べ替え重要度）。"""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=SEED,
        scoring="f1_macro", n_jobs=-1,
    )
    return {
        column: {
            "mean": float(result.importances_mean[index]),
            "std": float(result.importances_std[index]),
        }
        for index, column in enumerate(FEATURE_COLUMNS)
    }


# ---------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------

def main():
    print("1. 気象データを読み込み・検証中...")
    raw = load_dataset()
    fingerprint = data_module.dataset_fingerprint(DATA_PATH)
    print(f"   {len(raw)} 行 / {fingerprint['cities']}都市 / "
          f"{fingerprint['date_from']} 〜 {fingerprint['date_to']}")
    print(f"   データの指紋（sha256）: {fingerprint['sha256'][:16]}...")

    print("\n2. おすすめ度モデルで正解ラベルを作成中...")
    df = labeling.add_labels(raw)
    counts = labeling.label_distribution(df)
    print("   " + " / ".join(f"{name} {value}" for name, value in counts.items()))
    limit = labeling.bayes_accuracy(df)
    print(f"   理論上の正解率の上限（ベイズ限界）: {limit:.3f}")

    print("\n3. 学習用と評価用に分けています（8:2）...")
    X = df[FEATURE_COLUMNS]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG.train.test_size, random_state=SEED, stratify=y
    )
    frame_test = df.loc[X_test.index]
    print(f"   学習用: {len(X_train)} 件 / 評価用: {len(X_test)} 件")

    print(f"\n4. {CONFIG.train.cv_splits}分割の交差検証で、モデルを比べています...")
    comparison = compare_candidates(X_train, y_train)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    best_name = comparison.iloc[0]["モデル"]
    runner_up_name = comparison.iloc[1]["モデル"]
    print(f"\n   → いちばん成績が良かったモデル: {best_name}")

    print("\n5. 確率の質（較正）を確かめています...")
    model, calibration = calibrate_if_better(
        build_candidates()[best_name], X_train, y_train, X_test, y_test
    )
    print(f"   較正なし: ECE {calibration['before']['ece']:.4f} / "
          f"対数損失 {calibration['before']['log_loss']:.4f}")
    print(f"   較正あり: ECE {calibration['after']['ece']:.4f} / "
          f"対数損失 {calibration['after']['log_loss']:.4f}")
    print(f"   → {'較正あり' if calibration['calibrated'] else '較正なし'} を採用します")

    print("\n6. 評価用データで確かめています...")
    baseline = build_candidates()["ベースライン（最多クラス）"].fit(X_train, y_train)
    runner_up = build_candidates()[runner_up_name].fit(X_train, y_train)
    test_metrics = evaluate(
        model, X_test, y_test, frame_test,
        baseline.predict(X_test), runner_up.predict(X_test),
    )
    test_limit = labeling.bayes_accuracy(frame_test)

    accuracy = test_metrics["accuracy"]
    macro_f1 = test_metrics["macro_f1"]
    print(f"   正解率  : {accuracy['value']:.3f} "
          f"[95%信頼区間 {accuracy['low']:.3f}〜{accuracy['high']:.3f}]（上限 {test_limit:.3f}）")
    print(f"   マクロF1: {macro_f1['value']:.3f} "
          f"[{macro_f1['low']:.3f}〜{macro_f1['high']:.3f}]")
    print(f"   対数損失: {test_metrics['log_loss']:.3f} / "
          f"ECE {test_metrics['calibration']['ece']:.4f}")
    print(f"   ベースラインとの差: p={test_metrics['vs_baseline']['p_value']:.2e}"
          f"（{'有意' if test_metrics['vs_baseline']['significant'] else '有意でない'}）")
    print(f"   2位（{runner_up_name}）との差: p={test_metrics['vs_runner_up']['p_value']:.3f}"
          f"（{'有意' if test_metrics['vs_runner_up']['significant'] else '有意でない'}）")

    print("\n   混同行列（縦：正解 / 横：予測）:")
    print(pd.DataFrame(test_metrics["confusion_matrix"],
                       index=CATEGORIES, columns=CATEGORIES).to_string())

    print("\n7. 苦手なところを探しています（都市別・季節別）...")
    for name, rows in test_metrics["slices"].items():
        worst, best = rows[0], rows[-1]
        print(f"   {name:<7} 最低 {worst['group']} {worst['accuracy']:.3f} / "
              f"最高 {best['group']} {best['accuracy']:.3f} "
              f"（差 {best['accuracy'] - worst['accuracy']:.3f}）")

    print("\n8. どの入力が効いているかを調べています...")
    importances = feature_importances(model, X_test, y_test)
    for column, value in sorted(importances.items(),
                                key=lambda item: item[1]["mean"], reverse=True):
        print(f"   {column:<17} {value['mean']:.3f} ± {value['std']:.3f}")

    print("\n9. 未来のデータ（学習中に一度も見ていない）で最終確認...")
    holdout = evaluate_holdout(model)
    if holdout is None:
        print("   （data/weather_jp_holdout.csv が無いので省略しました）")
        print("   作るには: python fetch_weather.py --holdout")
    else:
        print(f"   期間: {holdout['date_from']} 〜 {holdout['date_to']}"
              f"（{holdout['rows']} 件）")
        print(f"   正解率  : {holdout['accuracy']['value']:.3f} "
              f"[{holdout['accuracy']['low']:.3f}〜{holdout['accuracy']['high']:.3f}]"
              f"（上限 {holdout['bayes_accuracy']:.3f}）")
        print(f"   マクロF1: {holdout['macro_f1']['value']:.3f} / "
              f"ECE {holdout['calibration']['ece']:.4f}")
        drop = accuracy["value"] - holdout["accuracy"]["value"]
        print(f"   評価用データからの落ちこみ: {drop:+.3f}")

    print("\n10. 全データで学習し直して保存します...")
    from sklearn.base import clone

    if calibration["calibrated"]:
        final_model = CalibratedClassifierCV(
            clone(build_candidates()[best_name]), method="isotonic",
            cv=CONFIG.train.cv_splits,
        ).fit(X, y)
    else:
        final_model = clone(build_candidates()[best_name]).fit(X, y)

    feature_ranges = {
        column: {
            "min": float(raw[column].min()),
            "max": float(raw[column].max()),
            "mean": float(raw[column].mean()),
        }
        for column in FEATURE_COLUMNS
    }

    card = {
        "model_name": MODEL_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "天気からお出かけカテゴリを当てる3クラス分類",
        "selected_model": best_name,
        "runner_up": runner_up_name,
        "estimator": type(final_model).__name__,
        "features": FEATURE_COLUMNS,
        "classes": CATEGORIES,
        "data": fingerprint,
        "dataset": {"label_counts": counts, "feature_ranges": feature_ranges},
        "labeling": {
            "method": "おすすめ度モデル（ルール）＋ソフトマックス抽選による弱教師あり",
            "softmax_temperature": CONFIG.label.softmax_temperature,
            "bayes_accuracy_all": limit,
            "bayes_accuracy_test": test_limit,
        },
        "training": {
            "test_size": CONFIG.train.test_size,
            "cv_splits": CONFIG.train.cv_splits,
            "random_seed": SEED,
            "final_fit": "全データで学習し直し",
        },
        "cv_comparison": comparison.to_dict(orient="records"),
        "calibration": calibration,
        "test_metrics": test_metrics,
        "holdout_metrics": holdout,
        "permutation_importance": importances,
    }

    bundle = ModelBundle(
        estimator=final_model,
        feature_names=FEATURE_COLUMNS,
        model_name=MODEL_NAME,
        version="",  # 履歴に記録したあとで入れる
        task=card["task"],
        classes=CATEGORIES,
        metadata={
            "created_at": card["created_at"],
            "data": fingerprint,
            "feature_ranges": feature_ranges,
            "selected_model": best_name,
            "calibrated": calibration["calibrated"],
        },
    )

    entry = Registry().record(
        model_name=MODEL_NAME,
        artifact_path=MODEL_PATH,
        task=card["task"],
        metrics={
            "test_accuracy": accuracy["value"],
            "test_macro_f1": macro_f1["value"],
            "test_log_loss": test_metrics["log_loss"],
            "test_ece": test_metrics["calibration"]["ece"],
            "holdout_accuracy": holdout["accuracy"]["value"] if holdout else None,
            "bayes_limit": test_limit,
        },
        data_fingerprint=fingerprint,
        params={"selected_model": best_name, "calibrated": calibration["calibrated"]},
    )

    bundle.version = entry["version"]
    card["version"] = entry["version"]
    card["git_sha"] = entry["git_sha"]
    card["environment"] = entry["environment"]

    save_bundle(MODEL_PATH, bundle)
    print(f"   モデルを保存しました: {MODEL_PATH}（{entry['version']}）")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(CARD_PATH, "w", encoding="utf-8") as file:
        json.dump(card, file, ensure_ascii=False, indent=2, default=str)
    print(f"   モデルカードを保存しました: {CARD_PATH}")

    print("\n完了！ 説明は doc/README.md にあります。")


if __name__ == "__main__":
    main()
