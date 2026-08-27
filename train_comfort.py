"""
お出かけプランナー：おでかけ日和度（快適度スコア）モデルの学習スクリプト

天気4項目から、その日の「おでかけ日和度」を 0〜100点で予測する回帰モデルを作ります。
カテゴリ予測が「どこへ行くか」を決めるのに対して、こちらは「どのくらい良い日か」を数字で表します。

分類ではなく回帰なので、成績は正解率ではなく MAE（平均どれくらいズレたか）で見ます。
点数にはわざと「人による感じ方のちがい」を入れてあるため、
どんなモデルでも越えられない誤差の下限があります。そこと比べて読んでください。

実行方法:
    python train_comfort.py
"""

import json
import os
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeRegressor

import train_model
from outing_ml import data as data_module
from outing_ml import labeling, metrics
from outing_ml.config import CONFIG, FEATURE_COLUMNS
from outing_ml.registry import ModelBundle, Registry, save_bundle

MODEL_NAME = "comfort-regressor"
SEED = CONFIG.train.random_seed

DATA_PATH = CONFIG.paths.dataset
MODEL_DIR = CONFIG.paths.model_dir
MODEL_PATH = CONFIG.paths.comfort_model
CARD_PATH = CONFIG.paths.comfort_card


def build_candidates():
    """比べるモデルの一覧を作る。"""
    return {
        "ベースライン（平均点を答える）": DummyRegressor(strategy="mean"),
        "線形回帰": LinearRegression(),
        "決定木（深さ8）": DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=20, random_state=SEED
        ),
        "ランダムフォレスト": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED
        ),
        "勾配ブースティング": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=SEED,
        ),
    }


def compare_candidates(X_train, y_train):
    """交差検証で候補を比べて、成績の表（DataFrame）を返す。"""
    cv = KFold(n_splits=CONFIG.train.cv_splits, shuffle=True, random_state=SEED)
    rows = []

    for name, model in build_candidates().items():
        scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
        )
        rows.append(
            {
                "モデル": name,
                "MAE": -scores["test_neg_mean_absolute_error"].mean(),
                "RMSE": -scores["test_neg_root_mean_squared_error"].mean(),
                "決定係数R2": scores["test_r2"].mean(),
                "MAEの標準偏差": scores["test_neg_mean_absolute_error"].std(),
            }
        )

    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def noise_floor(df):
    """人による感じ方のちがいのぶんだけ残る、誤差の下限。"""
    return {
        "mae": float(mean_absolute_error(df["comfort_score"], df["true_score"])),
        "rmse": float(np.sqrt(mean_squared_error(df["comfort_score"], df["true_score"]))),
        "r2": float(r2_score(df["comfort_score"], df["true_score"])),
    }


def residual_report(y_true, y_pred, bands=((0, 40), (40, 60), (60, 80), (80, 101))):
    """点数の帯ごとに、ズレの向きと大きさを見る。

    全体の MAE が小さくても、「低い点数の日だけ高めに出す」といった
    かたよりがあると、使う側は判断を誤ります。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rows = []

    for low, high in bands:
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() < 30:
            continue
        rows.append(
            {
                "band": f"{low}〜{high - 1}点",
                "count": int(mask.sum()),
                "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
                "bias": float((y_pred[mask] - y_true[mask]).mean()),
            }
        )

    return rows


def evaluate(model, X_test, y_test, frame_test):
    """評価用データでの成績（信頼区間・内訳つき）を返す。"""
    predicted = model.predict(X_test)
    errors = np.abs(predicted - np.asarray(y_test, dtype=float))

    result = metrics.regression_metrics(y_test, predicted)
    result["within_5_points"] = float((errors <= 5).mean())
    result["within_10_points"] = float((errors <= 10).mean())
    result["max_error"] = float(errors.max())
    result["residuals_by_band"] = residual_report(y_test, predicted)

    seasons = metrics.season_of(frame_test["date"])
    result["slices"] = {
        "city": _regression_slices(frame_test["city"], y_test, predicted),
        "season": _regression_slices(seasons, y_test, predicted),
    }
    return result


def _regression_slices(groups, y_true, y_pred, min_count=30):
    """グループごとの MAE を出す。"""
    frame = pd.DataFrame(
        {"group": np.asarray(groups), "true": np.asarray(y_true, dtype=float),
         "pred": np.asarray(y_pred, dtype=float)}
    )
    rows = []
    for group, part in frame.groupby("group"):
        if len(part) < min_count:
            continue
        rows.append(
            {
                "group": str(group),
                "count": int(len(part)),
                "mae": float(mean_absolute_error(part["true"], part["pred"])),
                "bias": float((part["pred"] - part["true"]).mean()),
            }
        )
    return sorted(rows, key=lambda row: row["mae"], reverse=True)


def evaluate_holdout(model, path=None):
    """学習中に一度も見ていない未来のデータで最終確認する。"""
    path = path or CONFIG.paths.holdout
    if not os.path.exists(path):
        return None

    frame = data_module.load_dataset(path)
    scored = labeling.add_comfort_scores(frame)
    predicted = model.predict(scored[FEATURE_COLUMNS])

    result = metrics.regression_metrics(scored["comfort_score"], predicted)
    result["rows"] = int(len(scored))
    result["date_from"] = str(frame["date"].min())
    result["date_to"] = str(frame["date"].max())
    result["noise_floor_mae"] = float(
        mean_absolute_error(scored["comfort_score"], scored["true_score"])
    )
    return result


def main():
    print("1. 気象データを読み込み・検証中...")
    raw = train_model.load_dataset()
    fingerprint = data_module.dataset_fingerprint(DATA_PATH)
    print(f"   {len(raw)} 行 / 指紋 {fingerprint['sha256'][:16]}...")

    print("\n2. おでかけ日和度（正解の点数）を計算中...")
    df = labeling.add_comfort_scores(raw)
    print(f"   平均 {df['comfort_score'].mean():.1f} 点 / "
          f"標準偏差 {df['comfort_score'].std():.1f}")
    print(f"   80点以上 {(df['comfort_score'] >= 80).mean() * 100:.1f}% / "
          f"40点未満 {(df['comfort_score'] < 40).mean() * 100:.1f}%")
    floor = noise_floor(df)
    print(f"   理論上の下限（これ以上は縮められない誤差）: MAE {floor['mae']:.2f} 点")

    print("\n3. 学習用と評価用に分けています（8:2）...")
    X = df[FEATURE_COLUMNS]
    y = df["comfort_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG.train.test_size, random_state=SEED
    )
    frame_test = df.loc[X_test.index]
    print(f"   学習用: {len(X_train)} 件 / 評価用: {len(X_test)} 件")

    print(f"\n4. {CONFIG.train.cv_splits}分割の交差検証で、モデルを比べています...")
    comparison = compare_candidates(X_train, y_train)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    best_name = comparison.iloc[0]["モデル"]
    print(f"\n   → いちばん成績が良かったモデル: {best_name}")

    print("\n5. 評価用データで確かめています...")
    model = clone(build_candidates()[best_name]).fit(X_train, y_train)
    test_metrics = evaluate(model, X_test, y_test, frame_test)
    print(f"   MAE       : {test_metrics['mae']:.2f} 点 "
          f"[95%信頼区間 {test_metrics['mae_low']:.2f}〜{test_metrics['mae_high']:.2f}]"
          f"（下限 {floor['mae']:.2f} 点）")
    print(f"   RMSE      : {test_metrics['rmse']:.2f} 点 / R2 {test_metrics['r2']:.3f}")
    print(f"   誤差5点以内: {test_metrics['within_5_points'] * 100:.1f}% / "
          f"10点以内: {test_metrics['within_10_points'] * 100:.1f}%")

    print("\n   点数の帯ごとのズレ（bias が＋なら高めに出している）:")
    for row in test_metrics["residuals_by_band"]:
        print(f"   {row['band']:<10} {row['count']:>5}件  MAE {row['mae']:5.2f}  "
              f"bias {row['bias']:+5.2f}")

    print("\n6. 苦手なところを探しています...")
    for name, rows in test_metrics["slices"].items():
        worst, best = rows[0], rows[-1]
        print(f"   {name:<7} 最悪 {worst['group']} MAE {worst['mae']:.2f} / "
              f"最良 {best['group']} MAE {best['mae']:.2f}")

    print("\n7. 未来のデータで最終確認...")
    holdout = evaluate_holdout(model)
    if holdout is None:
        print("   （data/weather_jp_holdout.csv が無いので省略しました）")
    else:
        print(f"   期間: {holdout['date_from']} 〜 {holdout['date_to']}"
              f"（{holdout['rows']} 件）")
        print(f"   MAE {holdout['mae']:.2f} 点（下限 {holdout['noise_floor_mae']:.2f}）/ "
              f"R2 {holdout['r2']:.3f}")
        print(f"   評価用データからの悪化: {holdout['mae'] - test_metrics['mae']:+.2f} 点")

    print("\n8. 全データで学習し直して保存します...")
    final_model = clone(build_candidates()[best_name]).fit(X, y)

    examples = [(22, 10, 2, 50), (15, 80, 4, 70), (33, 0, 2, 85), (5, 0, 8, 40)]
    predicted = final_model.predict(pd.DataFrame(examples, columns=FEATURE_COLUMNS))
    print("\n   予測の例:")
    for values, score in zip(examples, predicted, strict=True):
        print(f"   気温{values[0]:>4}℃ 降水{values[1]:>4}% 風速{values[2]:>3}m/s "
              f"湿度{values[3]:>3}% → {score:5.1f} 点")

    card = {
        "model_name": MODEL_NAME,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "task": "おでかけ日和度（0〜100点）を当てる回帰",
        "selected_model": best_name,
        "estimator": type(final_model).__name__,
        "features": FEATURE_COLUMNS,
        "target": "comfort_score",
        "score_noise": CONFIG.comfort.noise_std,
        "noise_floor": floor,
        "data": fingerprint,
        "dataset": {
            "rows": int(len(df)),
            "score_mean": float(df["comfort_score"].mean()),
            "score_std": float(df["comfort_score"].std()),
        },
        "training": {
            "test_size": CONFIG.train.test_size,
            "cv_splits": CONFIG.train.cv_splits,
            "random_seed": SEED,
        },
        "cv_comparison": comparison.to_dict(orient="records"),
        "test_metrics": test_metrics,
        "holdout_metrics": holdout,
        "examples": [
            {"input": dict(zip(FEATURE_COLUMNS, values, strict=True)), "score": float(score)}
            for values, score in zip(examples, predicted, strict=True)
        ],
    }

    entry = Registry().record(
        model_name=MODEL_NAME,
        artifact_path=MODEL_PATH,
        task=card["task"],
        metrics={
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "holdout_mae": holdout["mae"] if holdout else None,
            "noise_floor_mae": floor["mae"],
        },
        data_fingerprint=fingerprint,
        params={"selected_model": best_name},
    )

    card["version"] = entry["version"]
    card["git_sha"] = entry["git_sha"]
    card["environment"] = entry["environment"]

    save_bundle(
        MODEL_PATH,
        ModelBundle(
            estimator=final_model,
            feature_names=FEATURE_COLUMNS,
            model_name=MODEL_NAME,
            version=entry["version"],
            task=card["task"],
            target="comfort_score",
            metadata={"created_at": card["created_at"], "data": fingerprint,
                      "noise_floor": floor},
        ),
    )
    print(f"\n   モデルを保存しました: {MODEL_PATH}（{entry['version']}）")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(CARD_PATH, "w", encoding="utf-8") as file:
        json.dump(card, file, ensure_ascii=False, indent=2, default=str)
    print(f"   成績表を保存しました: {CARD_PATH}")

    print("\n完了！ 説明は doc/comfort.md にあります。")


if __name__ == "__main__":
    main()
