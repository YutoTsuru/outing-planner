"""
お出かけプランナー：翌日の天気を予測するモデルの学習スクリプト

きのう・きょうの天気から、あしたの天気（気温・降水確率・風速・湿度）を
予測するモデルを学習して保存します。

このモデルがあると、天気予報APIを使わなくても
「あしたのお出かけプラン」を組み立てられます。
    あしたの天気を予測 → その天気を outing_model.pkl に渡す → あしたのおすすめが出る

時系列データなので、学習と評価の分け方に注意が必要です。
ランダムに分けると「未来を見てから過去を当てる」ことになってしまうため、
    学習：2019〜2023年 ／ 評価：2024年
のように、時間で区切って分けています。

実行方法:
    python train_forecast.py
"""

import json
import os
import platform
from datetime import date

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import train_model

# 乱数の種（毎回同じ結果になるように固定する）
RANDOM_SEED = 42

# 読み込み先・保存先
DATA_PATH = train_model.DATA_PATH
MODEL_DIR = train_model.MODEL_DIR
MODEL_PATH = os.path.join(MODEL_DIR, "forecast_model.pkl")
CARD_PATH = os.path.join(MODEL_DIR, "forecast_card.json")

# 予測する4項目（app.py の入力欄と同じ）
TARGET_COLUMNS = train_model.FEATURE_COLUMNS

# 何日前までさかのぼって特徴量にするか
LAG_DAYS = [1, 2]

# 評価に使う年（この年より前が学習用）
TEST_YEAR = 2024

# 予測値がとりうる範囲（アプリのスライダーと同じ）
VALUE_LIMITS = {
    "temperature": (-10.0, 40.0),
    "rain_probability": (0.0, 100.0),
    "wind_speed": (0.0, 20.0),
    "humidity": (0.0, 100.0),
}


# ---------------------------------------------------------------
# 1. 特徴量づくり
# ---------------------------------------------------------------

def build_features(raw):
    """1日ずつの気象データから、「あしたを当てる」ための表を作る。

    1行が「ある都市の、ある日」で、
      入力：その日・前日・前々日の天気＋季節＋都市
      正解：その次の日の天気
    という形にします。
    """
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    grouped = df.groupby("city", sort=False)
    features = pd.DataFrame({"city": df["city"], "date": df["date"]})

    for column in TARGET_COLUMNS:
        # きょうの値
        features[f"{column}_today"] = df[column]
        # 何日か前の値
        for lag in LAG_DAYS:
            features[f"{column}_lag{lag}"] = grouped[column].shift(lag)
        # 直近3日の平均（その季節らしさが出る）
        features[f"{column}_mean3"] = (
            grouped[column].rolling(3, min_periods=3).mean().reset_index(level=0, drop=True)
        )
        # きのうからの変化（下がり続けているのか、上がり続けているのか）
        features[f"{column}_diff"] = df[column] - grouped[column].shift(1)
        # あしたの値（これが正解）
        features[f"{column}_next"] = grouped[column].shift(-1)

    # 季節。1年でひと回りするので、角度に直して sin と cos にする
    day_of_year = df["date"].dt.dayofyear
    features["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # 前後の日がそろっていない行（各都市の最初の2日と最後の1日）は使えない
    return features.dropna().reset_index(drop=True)


def feature_columns(features):
    """入力に使う列の名前を返す（正解の列と日付をのぞいたもの）。"""
    return [
        column
        for column in features.columns
        if column != "date" and not column.endswith("_next")
    ]


def split_by_year(features):
    """時間で学習用と評価用に分ける（未来を先に見てしまわないように）。"""
    is_test = features["date"].dt.year >= TEST_YEAR
    return features[~is_test].reset_index(drop=True), features[is_test].reset_index(drop=True)


# ---------------------------------------------------------------
# 2. モデル
# ---------------------------------------------------------------

def build_model(input_columns):
    """都市名を0/1に直してから、4項目まとめて回帰するモデルを作る。"""
    category_columns = ["city"]
    number_columns = [column for column in input_columns if column not in category_columns]

    preprocessor = ColumnTransformer(
        [
            ("city", OneHotEncoder(handle_unknown="ignore"), category_columns),
            ("numbers", "passthrough", number_columns),
        ]
    )

    regressor = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_SEED,
        )
    )

    return Pipeline([("prepare", preprocessor), ("regressor", regressor)])


def clip_predictions(predicted):
    """予測値を、アプリで入力できる範囲におさめる。"""
    clipped = predicted.copy()
    for index, column in enumerate(TARGET_COLUMNS):
        low, high = VALUE_LIMITS[column]
        clipped[:, index] = np.clip(clipped[:, index], low, high)
    return clipped


# ---------------------------------------------------------------
# 3. くらべる相手（ベースライン）
# ---------------------------------------------------------------

def persistence_baseline(test):
    """「あしたも、きょうと同じ天気」と答えるだけの予報。

    天気予報の世界では「持続予報」と呼ばれ、意外と当たります。
    これに勝てなければ、機械学習を使う意味がありません。
    """
    return test[[f"{column}_today" for column in TARGET_COLUMNS]].to_numpy(dtype=float)


def climatology_baseline(train, test):
    """「その都市の、その月の平均」を答えるだけの予報（平年値）。"""
    reference = train.copy()
    reference["month"] = reference["date"].dt.month
    averages = reference.groupby(["city", "month"])[
        [f"{column}_next" for column in TARGET_COLUMNS]
    ].mean()

    keys = pd.MultiIndex.from_arrays([test["city"], test["date"].dt.month])
    return averages.reindex(keys).to_numpy(dtype=float)


# ---------------------------------------------------------------
# 4. 評価
# ---------------------------------------------------------------

def score_predictions(actual, predicted):
    """項目ごとに、平均絶対誤差（MAE）と二乗平均平方根誤差（RMSE）を出す。"""
    scores = {}
    for index, column in enumerate(TARGET_COLUMNS):
        scores[column] = {
            "mae": float(mean_absolute_error(actual[:, index], predicted[:, index])),
            "rmse": float(np.sqrt(mean_squared_error(actual[:, index], predicted[:, index]))),
        }
    return scores


def print_score_table(named_scores):
    """複数の予報のMAEを、1つの表にして表示する。"""
    table = pd.DataFrame(
        {name: {column: scores[column]["mae"] for column in TARGET_COLUMNS}
         for name, scores in named_scores.items()}
    )
    print(table.round(2).to_string())


def recommendation_agreement(weather):
    """予測した天気を使っても、同じおすすめが出るかを確かめる。

    お出かけプランナーが最後に見るのは天気の数値ではなく「おすすめカテゴリ」です。
    そこで、実測の天気から出したおすすめと、予測の天気から出したおすすめが
    どれくらい一致するかを測ります。
    """
    if not os.path.exists(train_model.MODEL_PATH):
        return None

    category_model = joblib.load(train_model.MODEL_PATH)
    results = {}

    truth = category_model.predict(
        pd.DataFrame(weather["actual"], columns=TARGET_COLUMNS)
    )
    for name, values in weather.items():
        if name == "actual":
            continue
        predicted = category_model.predict(pd.DataFrame(values, columns=TARGET_COLUMNS))
        results[name] = float((predicted == truth).mean())

    return results


# ---------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------

def main():
    print("1. 気象データを読み込み中...")
    raw = train_model.load_dataset()
    print(f"   データ件数: {len(raw)}")

    print("\n2. 「あしたを当てる」形に組み直しています...")
    features = build_features(raw)
    input_columns = feature_columns(features)
    print(f"   使える行: {len(features)} ／ 入力に使う列: {len(input_columns)}")
    print(f"   （きょう・{LAG_DAYS[0]}日前・{LAG_DAYS[-1]}日前の天気、3日平均、前日差、季節、都市）")

    print(f"\n3. 時間で分けています（学習：〜{TEST_YEAR - 1}年 ／ 評価：{TEST_YEAR}年）...")
    train, test = split_by_year(features)
    print(f"   学習用: {len(train)} 件 ／ 評価用: {len(test)} 件")

    target_names = [f"{column}_next" for column in TARGET_COLUMNS]
    X_train, y_train = train[input_columns], train[target_names].to_numpy(dtype=float)
    X_test, y_test = test[input_columns], test[target_names].to_numpy(dtype=float)

    print("\n4. モデルを学習中...")
    model = build_model(input_columns)
    model.fit(X_train, y_train)

    print("\n5. 評価用データ（2024年）で確かめています...")
    predicted = clip_predictions(model.predict(X_test))
    persistence = persistence_baseline(test)
    climatology = climatology_baseline(train, test)

    named_scores = {
        "このモデル": score_predictions(y_test, predicted),
        "持続予報（あした＝きょう）": score_predictions(y_test, persistence),
        "平年値（都市×月の平均）": score_predictions(y_test, climatology),
    }
    print("   平均絶対誤差（MAE。小さいほど良い）:")
    print_score_table(named_scores)

    improvement = {
        column: float(
            1
            - named_scores["このモデル"][column]["mae"]
            / named_scores["持続予報（あした＝きょう）"][column]["mae"]
        )
        for column in TARGET_COLUMNS
    }
    print("\n   持続予報からどれだけ誤差を減らせたか:")
    for column, value in improvement.items():
        print(f"   {column:<17} {value * 100:5.1f}% 改善")

    print("\n6. おすすめカテゴリまで通して確かめています...")
    agreement = recommendation_agreement(
        {
            "actual": y_test,
            "このモデル": predicted,
            "持続予報（あした＝きょう）": persistence,
            "平年値（都市×月の平均）": climatology,
        }
    )
    if agreement is None:
        print("   （model/outing_model.pkl が無いので省略しました）")
    else:
        for name, value in agreement.items():
            print(f"   {name:<22} 実測の天気と同じおすすめになった割合: {value:.3f}")

    print("\n7. 全期間で学習し直して保存します...")
    final_model = build_model(input_columns)
    final_model.fit(features[input_columns], features[target_names].to_numpy(dtype=float))
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": final_model, "input_columns": input_columns}, MODEL_PATH)
    print(f"モデルを保存しました: {MODEL_PATH}")

    card = {
        "model_name": "outing-planner-next-day-forecast",
        "created_at": date.today().isoformat(),
        "task": "翌日の気象4項目を当てる多出力回帰",
        "estimator": "MultiOutputRegressor(HistGradientBoostingRegressor)",
        "targets": TARGET_COLUMNS,
        "input_columns": input_columns,
        "lag_days": LAG_DAYS,
        "dataset": {
            "path": DATA_PATH,
            "rows_used": int(len(features)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "split": f"学習：〜{TEST_YEAR - 1}年 ／ 評価：{TEST_YEAR}年（時間で分割）",
        },
        "metrics": named_scores,
        "improvement_over_persistence": improvement,
        "recommendation_agreement": agreement,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    with open(CARD_PATH, "w", encoding="utf-8") as file:
        json.dump(card, file, ensure_ascii=False, indent=2)
    print(f"成績表を保存しました: {CARD_PATH}")

    print("\n完了！ 説明は doc/forecast.md にあります。")


if __name__ == "__main__":
    main()
