"""
お出かけプランナー：おでかけ日和度（快適度スコア）モデルの学習スクリプト

天気4項目から、その日の「おでかけ日和度」を 0〜100点で予測するモデルを
学習して保存します。カテゴリ予測（train_model.py）が
「どこへ行くか」を決めるのに対して、こちらは「どのくらい良い日か」を数字で表します。

    例：22℃・降水確率10%・風速2m/s・湿度50% → 91点

分類ではなく回帰（数値を当てる問題）なので、
成績は正解率ではなく、MAE（平均どれくらいズレたか）で見ます。

実行方法:
    python train_comfort.py
"""

import json
import os
import platform
from datetime import date

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeRegressor

import train_model

# 乱数の種
RANDOM_SEED = 42

# 読み込み先・保存先
DATA_PATH = train_model.DATA_PATH
MODEL_DIR = train_model.MODEL_DIR
MODEL_PATH = os.path.join(MODEL_DIR, "comfort_model.pkl")
CARD_PATH = os.path.join(MODEL_DIR, "comfort_card.json")

# 入力に使う列（カテゴリ予測モデルと同じ4つ）
FEATURE_COLUMNS = train_model.FEATURE_COLUMNS

# 評価用データの割合と、交差検証の分割数
TEST_SIZE = 0.2
CV_SPLITS = 5

# 人による感じ方のちがい（点数のばらつき）の大きさ
SCORE_NOISE = 6.0


# ---------------------------------------------------------------
# 1. 正解の点数を作る
# ---------------------------------------------------------------
#
# 気象データに「その日は何点か」という正解はありません。
# そこで、気象の指標（体感温度・不快指数）をもとに減点方式で点数を決めます。
#
# ただし、同じ天気でも「良い日だ」と感じる度合いは人によってちがいます。
# そこで、計算した点数にばらつき（正規分布のノイズ）を足したものを正解にします。
# このばらつきがあるぶん、どんなモデルでもピタリとは当てられません。

def true_comfort_score(df):
    """天気から「おでかけ日和度」の素点（0〜100）を計算する。"""
    temperature = df["temperature"].to_numpy(dtype=float)
    rain = df["rain_probability"].to_numpy(dtype=float)
    wind = df["wind_speed"].to_numpy(dtype=float)
    humidity = df["humidity"].to_numpy(dtype=float)

    felt = temperature - 1.5 * np.sqrt(wind)              # 体感温度
    discomfort = train_model.discomfort_index(temperature, humidity)  # 不快指数

    penalty = (
        1.8 * np.maximum(0.0, 22.0 - felt)        # 寒いほど減点
        + 2.4 * np.maximum(0.0, temperature - 26.0)  # 暑いほど大きく減点
        + 0.45 * rain                              # 雨は重い減点
        + 1.6 * np.maximum(0.0, wind - 3.0)        # 風が強いと減点
        + 1.2 * np.maximum(0.0, discomfort - 75.0)  # 蒸し暑いと減点
    )

    return np.clip(100.0 - penalty, 0.0, 100.0)


def add_scores(df):
    """DataFrame に、素点と、ばらつきを足した正解の点数を足す。"""
    rng = np.random.default_rng(RANDOM_SEED)
    true_score = true_comfort_score(df)
    observed = np.clip(true_score + rng.normal(0.0, SCORE_NOISE, len(df)), 0.0, 100.0)

    df = df.copy()
    df["true_score"] = true_score
    df["comfort_score"] = observed
    return df


def noise_limit(df):
    """「これ以上は当てられない」という誤差の下限を計算する。

    正解にばらつきを足しているので、素点をそのまま答えても誤差は残ります。
    その残り（＝どんなモデルでも避けられない誤差）が下限です。
    """
    return {
        "mae": float(mean_absolute_error(df["comfort_score"], df["true_score"])),
        "rmse": float(np.sqrt(mean_squared_error(df["comfort_score"], df["true_score"]))),
        "r2": float(r2_score(df["comfort_score"], df["true_score"])),
    }


# ---------------------------------------------------------------
# 2. モデルの候補
# ---------------------------------------------------------------

def build_candidates():
    """比べるモデルの一覧を作る。"""
    return {
        "ベースライン（平均点を答える）": DummyRegressor(strategy="mean"),
        "線形回帰": LinearRegression(),
        "決定木（深さ8）": DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=20, random_state=RANDOM_SEED
        ),
        "ランダムフォレスト": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_SEED
        ),
        "勾配ブースティング": HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_SEED,
        ),
    }


def compare_candidates(X_train, y_train):
    """交差検証で候補を比べて、成績の表（DataFrame）を返す。"""
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    rows = []

    for name, model in build_candidates().items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
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


# ---------------------------------------------------------------
# 3. 評価
# ---------------------------------------------------------------

def evaluate(model, X_test, y_test):
    """評価用データでの成績をまとめて返す。"""
    predicted = model.predict(X_test)
    errors = np.abs(predicted - y_test)

    return {
        "mae": float(mean_absolute_error(y_test, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "r2": float(r2_score(y_test, predicted)),
        "within_5_points": float((errors <= 5).mean()),
        "within_10_points": float((errors <= 10).mean()),
        "max_error": float(errors.max()),
    }


# ---------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------

def main():
    print("1. 気象データを読み込み中...")
    raw = train_model.load_dataset()
    print(f"   データ件数: {len(raw)}")

    print("\n2. おでかけ日和度（正解の点数）を計算中...")
    df = add_scores(raw)
    print(f"   点数の平均: {df['comfort_score'].mean():.1f} 点 "
          f"／ 標準偏差: {df['comfort_score'].std():.1f}")
    print(f"   80点以上の日: {(df['comfort_score'] >= 80).mean() * 100:.1f}% "
          f"／ 40点未満の日: {(df['comfort_score'] < 40).mean() * 100:.1f}%")
    limit = noise_limit(df)
    print(f"   理論上の下限（これ以上は縮められない誤差）: MAE {limit['mae']:.2f} 点")

    print("\n3. 学習用と評価用に分けています（8:2）...")
    X = df[FEATURE_COLUMNS]
    y = df["comfort_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    print(f"   学習用: {len(X_train)} 件 ／ 評価用: {len(X_test)} 件")

    print(f"\n4. {CV_SPLITS}分割の交差検証で、モデルを比べています...")
    comparison = compare_candidates(X_train, y_train)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    best_name = comparison.iloc[0]["モデル"]
    print(f"\n   → いちばん成績が良かったモデル: {best_name}")

    print("\n5. 選んだモデルを学習して、評価用データで確かめています...")
    best_model = build_candidates()[best_name]
    best_model.fit(X_train, y_train)
    metrics = evaluate(best_model, X_test, y_test)
    print(f"   MAE       : {metrics['mae']:.2f} 点（下限 {limit['mae']:.2f} 点）")
    print(f"   RMSE      : {metrics['rmse']:.2f} 点")
    print(f"   決定係数R2: {metrics['r2']:.3f}")
    print(f"   誤差5点以内: {metrics['within_5_points'] * 100:.1f}% "
          f"／ 10点以内: {metrics['within_10_points'] * 100:.1f}%")

    print("\n6. 全データで学習し直して保存します...")
    final_model = build_candidates()[best_name]
    final_model.fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"モデルを保存しました: {MODEL_PATH}")

    examples = [(22, 10, 2, 50), (15, 80, 4, 70), (33, 0, 2, 85), (5, 0, 8, 40)]
    predicted = final_model.predict(pd.DataFrame(examples, columns=FEATURE_COLUMNS))
    print("\n   予測の例:")
    for values, score in zip(examples, predicted):
        print(f"   気温{values[0]:>4}℃ 降水{values[1]:>4}% 風速{values[2]:>3}m/s "
              f"湿度{values[3]:>3}% → {score:5.1f} 点")

    card = {
        "model_name": "outing-planner-comfort-score",
        "created_at": date.today().isoformat(),
        "task": "おでかけ日和度（0〜100点）を当てる回帰",
        "selected_model": best_name,
        "estimator": type(final_model).__name__,
        "features": FEATURE_COLUMNS,
        "target": "comfort_score",
        "score_noise": SCORE_NOISE,
        "noise_limit": limit,
        "dataset": {
            "path": DATA_PATH,
            "rows": int(len(df)),
            "score_mean": float(df["comfort_score"].mean()),
            "score_std": float(df["comfort_score"].std()),
        },
        "training": {
            "test_size": TEST_SIZE,
            "cv_splits": CV_SPLITS,
            "random_seed": RANDOM_SEED,
        },
        "cv_comparison": comparison.to_dict(orient="records"),
        "test_metrics": metrics,
        "examples": [
            {"input": dict(zip(FEATURE_COLUMNS, values)), "score": float(score)}
            for values, score in zip(examples, predicted)
        ],
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    with open(CARD_PATH, "w", encoding="utf-8") as file:
        json.dump(card, file, ensure_ascii=False, indent=2)
    print(f"\n成績表を保存しました: {CARD_PATH}")

    print("\n完了！ 説明は doc/comfort.md にあります。")


if __name__ == "__main__":
    main()
