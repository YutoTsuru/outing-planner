"""
お出かけプランナー：翌日の天気を予測するモデルの学習スクリプト

きのう・きょうの天気から、あしたの天気（気温・降水確率・風速・湿度）を予測します。

時系列の予測モデルで、いちばんやってはいけないのが
「1回だけ時間で区切って、良い数字が出たので採用」です。
たまたまその年が当てやすかっただけかもしれません。

そこで本番と同じ順番で何度も検証します（ウォークフォワード検証）。
    2020年までで学習 → 2021年を予測
    2021年までで学習 → 2022年を予測
    ...
さらに、点だけでなく予測区間（ここからここまでに入りそう）も作り、
その区間に実際の値がどれだけ入ったか（カバレッジ）まで確かめます。

実行方法:
    python train_forecast.py
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import train_model
from outing_ml import data as data_module
from outing_ml import features as feature_module
from outing_ml import metrics
from outing_ml.config import CONFIG, FEATURE_COLUMNS, INPUT_RANGES
from outing_ml.registry import ModelBundle, Registry, load_bundle, save_bundle

MODEL_NAME = "next-day-forecast"
SEED = CONFIG.train.random_seed

DATA_PATH = CONFIG.paths.dataset
MODEL_DIR = CONFIG.paths.model_dir
MODEL_PATH = CONFIG.paths.forecast_model
CARD_PATH = CONFIG.paths.forecast_card

TARGET_COLUMNS = FEATURE_COLUMNS


# ---------------------------------------------------------------
# モデル
# ---------------------------------------------------------------

def _preprocessor(input_columns):
    """都市名だけを 0/1 に展開し、それ以外はそのまま通す。"""
    number_columns = [column for column in input_columns if column != "city"]
    return ColumnTransformer(
        [
            ("city", OneHotEncoder(handle_unknown="ignore"), ["city"]),
            ("numbers", "passthrough", number_columns),
        ]
    )


def build_point_model(input_columns):
    """4項目を同時に予測する（真ん中の値をあてるモデル）。"""
    return Pipeline(
        [
            ("prepare", _preprocessor(input_columns)),
            (
                "regressor",
                MultiOutputRegressor(
                    HistGradientBoostingRegressor(
                        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                        l2_regularization=1.0, random_state=SEED,
                    )
                ),
            ),
        ]
    )


def build_quantile_model(input_columns, quantile):
    """「たぶんこれ以下」「たぶんこれ以上」を当てるモデル（分位点回帰）。

    点で1つ答えるだけだと、外れたときにどれくらい外れるのかが分かりません。
    上下の線を引いておくと、「だいたいこの範囲」と伝えられます。
    """
    return Pipeline(
        [
            ("prepare", _preprocessor(input_columns)),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    loss="quantile", quantile=quantile,
                    max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                    l2_regularization=1.0, random_state=SEED,
                ),
            ),
        ]
    )


def clip_predictions(predicted):
    """予測値を、アプリで入力できる範囲におさめる。"""
    clipped = np.asarray(predicted, dtype=float).copy()
    for index, column in enumerate(TARGET_COLUMNS):
        low, high = INPUT_RANGES[column]
        clipped[:, index] = np.clip(clipped[:, index], low, high)
    return clipped


# ---------------------------------------------------------------
# ベースライン
# ---------------------------------------------------------------

def persistence_baseline(frame):
    """「あしたも、きょうと同じ天気」と答えるだけの予報（持続予報）。"""
    return frame[[f"{column}_today" for column in TARGET_COLUMNS]].to_numpy(dtype=float)


def climatology_baseline(train, test):
    """「その都市の、その月の平均」を答えるだけの予報（平年値）。"""
    reference = train.copy()
    reference["month"] = reference["date"].dt.month
    averages = reference.groupby(["city", "month"])[
        [f"{column}{feature_module.TARGET_SUFFIX}" for column in TARGET_COLUMNS]
    ].mean()

    keys = pd.MultiIndex.from_arrays([test["city"], test["date"].dt.month])
    return averages.reindex(keys).to_numpy(dtype=float)


def score_predictions(actual, predicted):
    """項目ごとに MAE と RMSE を出す。"""
    return {
        column: {
            "mae": float(np.mean(np.abs(actual[:, index] - predicted[:, index]))),
            "rmse": float(
                np.sqrt(np.mean((actual[:, index] - predicted[:, index]) ** 2))
            ),
        }
        for index, column in enumerate(TARGET_COLUMNS)
    }


def mae_table(named_scores):
    """複数の予報のMAEを1つの表にする。"""
    return pd.DataFrame(
        {
            name: {column: scores[column]["mae"] for column in TARGET_COLUMNS}
            for name, scores in named_scores.items()
        }
    )


# ---------------------------------------------------------------
# ウォークフォワード検証
# ---------------------------------------------------------------

def walk_forward(features, input_columns, target_names):
    """本番と同じ順番で、年ごとに学習と予測をくり返す。"""
    folds = []

    for year, train, test in feature_module.walk_forward_splits(features):
        model = build_point_model(input_columns)
        model.fit(train[input_columns], train[target_names].to_numpy(dtype=float))

        actual = test[target_names].to_numpy(dtype=float)
        predicted = clip_predictions(model.predict(test[input_columns]))

        fold = {
            "year": int(year),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "model": score_predictions(actual, predicted),
            "persistence": score_predictions(actual, persistence_baseline(test)),
            "climatology": score_predictions(actual, climatology_baseline(train, test)),
        }
        folds.append(fold)

        print(f"   {year}年（学習 {len(train)} 件 → 予測 {len(test)} 件）: "
              f"気温MAE {fold['model']['temperature']['mae']:.2f}"
              f"（持続予報 {fold['persistence']['temperature']['mae']:.2f}）")

    return folds


def summarize_folds(folds):
    """年ごとの結果を、平均とばらつきにまとめる。"""
    summary = {}

    for source in ("model", "persistence", "climatology"):
        summary[source] = {}
        for column in TARGET_COLUMNS:
            values = [fold[source][column]["mae"] for fold in folds]
            summary[source][column] = {
                "mae_mean": float(np.mean(values)),
                "mae_std": float(np.std(values)),
                "mae_worst": float(np.max(values)),
            }

    summary["improvement_over_persistence"] = {
        column: float(
            1
            - summary["model"][column]["mae_mean"]
            / summary["persistence"][column]["mae_mean"]
        )
        for column in TARGET_COLUMNS
    }
    return summary


# ---------------------------------------------------------------
# 予測区間
# ---------------------------------------------------------------

def fit_quantile_models(train, input_columns, target_names):
    """項目ごと・分位点ごとにモデルを学習する。"""
    models = {}

    for column, target in zip(TARGET_COLUMNS, target_names):
        models[column] = {}
        for quantile in CONFIG.forecast.quantiles:
            model = build_quantile_model(input_columns, quantile)
            model.fit(train[input_columns], train[target].to_numpy(dtype=float))
            models[column][quantile] = model

    return models


def evaluate_intervals(models, test, input_columns, target_names):
    """予測区間の良し悪しを、カバレッジとピンボール損失で測る。"""
    low_quantile = min(CONFIG.forecast.quantiles)
    high_quantile = max(CONFIG.forecast.quantiles)
    nominal = high_quantile - low_quantile

    report = {"nominal_coverage": float(nominal), "targets": {}}

    for column, target in zip(TARGET_COLUMNS, target_names):
        actual = test[target].to_numpy(dtype=float)
        lower = models[column][low_quantile].predict(test[input_columns])
        upper = models[column][high_quantile].predict(test[input_columns])
        # 下限が上限を超えることがある（別々に学習しているため）ので、必ず並べ替える
        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)

        report["targets"][column] = {
            "coverage": metrics.interval_coverage(actual, lower, upper),
            "mean_width": float(np.mean(upper - lower)),
            "pinball": {
                str(quantile): metrics.pinball_loss(
                    actual, models[column][quantile].predict(test[input_columns]), quantile
                )
                for quantile in CONFIG.forecast.quantiles
            },
        }

    return report


# ---------------------------------------------------------------
# おすすめカテゴリまで通した評価
# ---------------------------------------------------------------

def recommendation_agreement(weather):
    """予測した天気を使っても、同じおすすめが出るかを確かめる。"""
    try:
        bundle = load_bundle(CONFIG.paths.category_model)
    except FileNotFoundError:
        return None

    truth = bundle.estimator.predict(
        pd.DataFrame(weather["actual"], columns=TARGET_COLUMNS)
    )

    return {
        name: float(
            (bundle.estimator.predict(pd.DataFrame(values, columns=TARGET_COLUMNS)) == truth).mean()
        )
        for name, values in weather.items()
        if name != "actual"
    }


# ---------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------

def main():
    print("1. 気象データを読み込み・検証中...")
    raw = train_model.load_dataset()
    fingerprint = data_module.dataset_fingerprint(DATA_PATH)
    print(f"   {len(raw)} 行 / 指紋 {fingerprint['sha256'][:16]}...")

    print("\n2. 「あしたを当てる」形に組み直しています...")
    features = feature_module.build_supervised_frame(raw)
    feature_module.assert_no_leakage(features)
    input_columns = feature_module.input_columns(features)
    target_names = feature_module.target_columns()
    print(f"   使える行: {len(features)} / 入力に使う列: {len(input_columns)}")
    print("   （きょう・1日前・2日前の天気、3日平均、前日差、季節、都市）")
    print("   未来の情報がまざっていないことを確認しました")

    print("\n3. ウォークフォワード検証（本番と同じ順番で年ごとに検証）...")
    folds = walk_forward(features, input_columns, target_names)
    summary = summarize_folds(folds)

    print("\n   年をまたいだ平均MAE:")
    table = pd.DataFrame(
        {
            "このモデル": {c: summary["model"][c]["mae_mean"] for c in TARGET_COLUMNS},
            "年ごとのばらつき": {c: summary["model"][c]["mae_std"] for c in TARGET_COLUMNS},
            "持続予報": {c: summary["persistence"][c]["mae_mean"] for c in TARGET_COLUMNS},
            "平年値": {c: summary["climatology"][c]["mae_mean"] for c in TARGET_COLUMNS},
        }
    )
    print(table.round(2).to_string())

    print("\n   持続予報からの改善:")
    for column, value in summary["improvement_over_persistence"].items():
        print(f"   {column:<17} {value * 100:5.1f}%")

    print("\n4. 最終モデルを学習しています（2019〜2024年の全期間）...")
    model = build_point_model(input_columns)
    model.fit(features[input_columns], features[target_names].to_numpy(dtype=float))

    print("\n5. 予測区間（80%）のモデルを学習しています...")
    quantile_models = fit_quantile_models(features, input_columns, target_names)
    print(f"   {len(TARGET_COLUMNS)}項目 × {len(CONFIG.forecast.quantiles)}分位点 "
          f"= {len(TARGET_COLUMNS) * len(CONFIG.forecast.quantiles)} 個")

    print("\n6. 未来のデータ（2025年〜）で最終確認...")
    holdout = None
    if os.path.exists(CONFIG.paths.holdout):
        holdout_raw = data_module.load_dataset(CONFIG.paths.holdout)
        holdout_features = feature_module.build_supervised_frame(holdout_raw)
        actual = holdout_features[target_names].to_numpy(dtype=float)
        predicted = clip_predictions(model.predict(holdout_features[input_columns]))
        persistence = persistence_baseline(holdout_features)
        climatology = climatology_baseline(features, holdout_features)

        named = {
            "このモデル": score_predictions(actual, predicted),
            "持続予報": score_predictions(actual, persistence),
            "平年値": score_predictions(actual, climatology),
        }
        print(f"   期間: {holdout_features['date'].min().date()} 〜 "
              f"{holdout_features['date'].max().date()}（{len(holdout_features)} 件）")
        print("   平均絶対誤差（MAE）:")
        print(mae_table(named).round(2).to_string())

        intervals = evaluate_intervals(
            quantile_models, holdout_features, input_columns, target_names
        )
        print(f"\n   予測区間（想定 {intervals['nominal_coverage'] * 100:.0f}%）の実績:")
        for column, report in intervals["targets"].items():
            print(f"   {column:<17} 実際に入った割合 {report['coverage'] * 100:5.1f}% "
                  f"/ 区間の幅 {report['mean_width']:.1f}")

        agreement = recommendation_agreement(
            {
                "actual": actual,
                "このモデル": predicted,
                "持続予報": persistence,
                "平年値": climatology,
            }
        )
        if agreement:
            print("\n   おすすめカテゴリまで通したときの一致率:")
            for name, value in agreement.items():
                print(f"   {name:<10} {value:.3f}")

        holdout = {
            "rows": int(len(holdout_features)),
            "date_from": str(holdout_features["date"].min().date()),
            "date_to": str(holdout_features["date"].max().date()),
            "metrics": named,
            "intervals": intervals,
            "recommendation_agreement": agreement,
        }
    else:
        print("   （data/weather_jp_holdout.csv が無いので省略しました）")

    print("\n7. 保存します...")
    card = {
        "model_name": MODEL_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "翌日の気象4項目を当てる多出力回帰（予測区間つき）",
        "estimator": "MultiOutputRegressor(HistGradientBoostingRegressor)",
        "targets": TARGET_COLUMNS,
        "input_columns": input_columns,
        "lag_days": list(CONFIG.forecast.lag_days),
        "quantiles": list(CONFIG.forecast.quantiles),
        "data": fingerprint,
        "dataset": {
            "rows_used": int(len(features)),
            "validation": "ウォークフォワード検証（年ごと）",
        },
        "walk_forward": {"folds": folds, "summary": summary},
        "holdout": holdout,
    }

    entry = Registry().record(
        model_name=MODEL_NAME,
        artifact_path=MODEL_PATH,
        task=card["task"],
        metrics={
            "walk_forward_temperature_mae": summary["model"]["temperature"]["mae_mean"],
            "walk_forward_rain_mae": summary["model"]["rain_probability"]["mae_mean"],
            "improvement_over_persistence": summary["improvement_over_persistence"],
            "holdout_temperature_mae": (
                holdout["metrics"]["このモデル"]["temperature"]["mae"] if holdout else None
            ),
        },
        data_fingerprint=fingerprint,
        params={"lag_days": list(CONFIG.forecast.lag_days)},
    )

    card["version"] = entry["version"]
    card["git_sha"] = entry["git_sha"]
    card["environment"] = entry["environment"]

    save_bundle(
        MODEL_PATH,
        ModelBundle(
            estimator={"point": model, "quantiles": quantile_models},
            feature_names=input_columns,
            model_name=MODEL_NAME,
            version=entry["version"],
            task=card["task"],
            target=",".join(TARGET_COLUMNS),
            metadata={
                "created_at": card["created_at"],
                "data": fingerprint,
                "quantiles": list(CONFIG.forecast.quantiles),
                "targets": TARGET_COLUMNS,
            },
        ),
    )
    print(f"   モデルを保存しました: {MODEL_PATH}（{entry['version']}）")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(CARD_PATH, "w", encoding="utf-8") as file:
        json.dump(card, file, ensure_ascii=False, indent=2, default=str)
    print(f"   成績表を保存しました: {CARD_PATH}")

    print("\n完了！ 説明は doc/forecast.md にあります。")


if __name__ == "__main__":
    main()
