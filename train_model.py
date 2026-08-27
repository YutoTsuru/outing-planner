"""
お出かけプランナー：モデル学習スクリプト

実測の気象データ（気温・降水確率・風速・湿度）から、
おすすめのお出かけカテゴリ（outdoor / indoor / relax）を予測する
機械学習モデルを学習して保存します。

このスクリプトがやることは4つです。

    1. 気象データ（data/weather_jp.csv）を読み込む
       …無ければ Open-Meteo から自動でダウンロードします
    2. 「おすすめ度モデル」で1日ごとに正解ラベルを付ける
    3. 5つの候補（ものさし1つ＋機械学習4つ）を比べて、いちばん良いものを選ぶ
    4. 選んだモデルを学習して model/outing_model.pkl に保存する
       あわせて、成績表を model/model_card.json に書き出す

くわしい説明（どんなモデルなのか・成績・限界）は doc/README.md にあります。

実行方法:
    python train_model.py
"""

import json
import os
import platform
from datetime import date

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import fetch_weather

# 乱数の種（毎回まったく同じ結果になるように固定する）
RANDOM_SEED = 42

# 読み込み先・保存先
DATA_PATH = fetch_weather.DATA_PATH
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "outing_model.pkl")
CARD_PATH = os.path.join(MODEL_DIR, "model_card.json")

# 特徴量の列名（app.py 側と同じ順番で使うこと）
FEATURE_COLUMNS = ["temperature", "rain_probability", "wind_speed", "humidity"]

# 予測するカテゴリ（この順番で確率が並ぶ）
CATEGORIES = ["indoor", "outdoor", "relax"]

# 評価用データの割合
TEST_SIZE = 0.2

# 交差検証の分割数
CV_SPLITS = 5


# ---------------------------------------------------------------
# 1. 正解ラベルを作る（おすすめ度モデル）
# ---------------------------------------------------------------
#
# 気象データには「その日どこへ行くべきか」の正解は入っていません。
# そこで、天気から3カテゴリの「おすすめ度」を計算するルールを決めて、
# それを正解ラベルのかわりにします（弱教師あり学習と呼ばれる作り方）。
#
# ルールをそのまま使うと正解が1つに決まってしまい、モデルは
# ルールを丸暗記するだけになります。そこで、おすすめ度を確率に直してから
# ラベルを抽選します。こうすると「どちらとも言える日」がまざるので、
# 実際の予測に近い、歯ごたえのある問題になります。

# 抽選のばらつき。小さいほど迷いがなくなり、大きいほど迷う日が増える
SOFTMAX_TEMPERATURE = 0.6


def discomfort_index(temperature, humidity):
    """不快指数。気温と湿度から、蒸し暑さを表す指標を計算する。"""
    return 0.81 * temperature + 0.01 * humidity * (0.99 * temperature - 14.3) + 46.3


def outing_scores(df):
    """1日ごとに、3カテゴリの「おすすめ度」を計算して返す。"""
    temperature = df["temperature"].to_numpy(dtype=float)
    rain = df["rain_probability"].to_numpy(dtype=float)
    wind = df["wind_speed"].to_numpy(dtype=float)
    humidity = df["humidity"].to_numpy(dtype=float)

    # 風があると、同じ気温でも寒く感じる（体感温度）
    felt_cold = temperature - 1.5 * np.sqrt(wind)
    discomfort = discomfort_index(temperature, humidity)

    # 屋外：22℃前後がいちばん快適。雨・風・蒸し暑さで下がる
    outdoor = (
        3.6 * np.exp(-((temperature - 22.0) ** 2) / (2 * 8.5 ** 2))
        - 0.048 * rain
        - 0.200 * wind
        - 0.020 * np.maximum(0.0, humidity - 70.0)
    )

    # 屋内：雨の日と、風が強い日（4m/s をこえたぶん）に上がる
    indoor = (
        0.35
        + 0.040 * rain
        + 0.220 * np.maximum(0.0, wind - 4.0)
    )

    # リラックス：蒸し暑い日・寒い日・真夏日ほど上がる（雨の日は屋内にゆずる）
    relax = (
        0.80
        + 0.050 * np.maximum(0.0, humidity - 72.0)
        + 0.150 * np.maximum(0.0, 13.0 - felt_cold)
        + 0.160 * np.maximum(0.0, temperature - 27.0)
        + 0.025 * np.maximum(0.0, discomfort - 76.0)
        - 0.008 * rain
    )

    # CATEGORIES と同じ並び順にそろえる
    return np.column_stack([indoor, outdoor, relax])


def scores_to_probabilities(scores):
    """おすすめ度を、合計が1になる確率に変換する（ソフトマックス）。"""
    scaled = scores / SOFTMAX_TEMPERATURE
    # 大きな数の指数計算で桁があふれないよう、行ごとに最大値を引いてから計算する
    exponent = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exponent / exponent.sum(axis=1, keepdims=True)


def add_labels(df):
    """DataFrame に、正解ラベルの列と、その日の「本当の確率」の列を足す。"""
    probabilities = scores_to_probabilities(outing_scores(df))
    rng = np.random.default_rng(RANDOM_SEED)

    # 行ごとに、確率にしたがってカテゴリを1つ抽選する
    draws = rng.random(len(df))
    cumulative = probabilities.cumsum(axis=1)
    chosen = (draws[:, None] > cumulative).sum(axis=1)

    df = df.copy()
    df["label"] = [CATEGORIES[i] for i in chosen]
    for index, name in enumerate(CATEGORIES):
        df[f"true_prob_{name}"] = probabilities[:, index]
    return df


def bayes_accuracy(df):
    """「これ以上は当てられない」という正解率の上限を計算する。

    ラベルは確率で抽選しているので、どんなに賢いモデルでも
    いちばん確率の高いカテゴリを答えるのが精一杯です。その平均値が上限になります。
    """
    columns = [f"true_prob_{name}" for name in CATEGORIES]
    return float(df[columns].to_numpy().max(axis=1).mean())


# ---------------------------------------------------------------
# 2. データを読み込む
# ---------------------------------------------------------------

def load_dataset():
    """気象データを読み込む（無ければダウンロードする）。"""
    if not os.path.exists(DATA_PATH):
        print(f"   {DATA_PATH} が無いので、先にダウンロードします")
        df = fetch_weather.download_all()
        os.makedirs(fetch_weather.DATA_DIR, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


# ---------------------------------------------------------------
# 3. モデルの候補
# ---------------------------------------------------------------

def build_candidates():
    """比べるモデルの一覧を作る。

    まず「いちばん多いカテゴリを答えるだけ」のものさし（ベースライン）を置き、
    そこからどれだけ良くなったかで、機械学習の効果を確かめます。
    """
    return {
        "ベースライン（最多クラス）": DummyClassifier(
            strategy="prior", random_state=RANDOM_SEED
        ),
        "ロジスティック回帰": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
            ]
        ),
        "決定木（深さ6）": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, random_state=RANDOM_SEED
        ),
        "ランダムフォレスト": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "勾配ブースティング": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_SEED,
        ),
    }


def compare_candidates(X_train, y_train):
    """交差検証で候補を比べて、成績の表（DataFrame）を返す。"""
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    rows = []

    for name, model in build_candidates().items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "f1_macro", "neg_log_loss"],
            n_jobs=None,
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
# 4. 評価
# ---------------------------------------------------------------

def evaluate(model, X_test, y_test):
    """評価用データでの成績をまとめて返す。"""
    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro")),
        "log_loss": float(log_loss(y_test, probabilities, labels=list(model.classes_))),
        "confusion_matrix": confusion_matrix(
            y_test, predicted, labels=CATEGORIES
        ).tolist(),
        "per_class": classification_report(
            y_test, predicted, labels=CATEGORIES, output_dict=True, zero_division=0
        ),
    }


def feature_importances(model, X_test, y_test):
    """どの入力が予測に効いているかを調べる（並べ替え重要度）。

    ある列の値をシャッフルして、成績がどれだけ落ちるかを見る方法です。
    どんなモデルにも同じ手順で使えます。
    """
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_SEED,
        scoring="f1_macro",
        n_jobs=-1,
    )
    return {
        column: {
            "mean": float(result.importances_mean[index]),
            "std": float(result.importances_std[index]),
        }
        for index, column in enumerate(FEATURE_COLUMNS)
    }


# ---------------------------------------------------------------
# 5. 保存
# ---------------------------------------------------------------

def save_model(model):
    """学習済みモデルをファイルに保存する。"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"モデルを保存しました: {MODEL_PATH}")


def save_model_card(card):
    """成績や設定を JSON にまとめて保存する（doc/README.md のもとになる情報）。"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(CARD_PATH, "w", encoding="utf-8") as file:
        json.dump(card, file, ensure_ascii=False, indent=2)
    print(f"モデルカードを保存しました: {CARD_PATH}")


# ---------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------

def main():
    print("1. 気象データを読み込み中...")
    raw = load_dataset()
    print(f"   データ件数: {len(raw)}（{raw['city'].nunique()}都市 / "
          f"{raw['date'].min()} 〜 {raw['date'].max()}）")

    print("\n2. おすすめ度モデルで正解ラベルを作成中...")
    df = add_labels(raw)
    label_counts = df["label"].value_counts()
    print("   カテゴリごとの件数:")
    print(label_counts.to_string())
    limit = bayes_accuracy(df)
    print(f"   理論上の正解率の上限（ベイズ限界）: {limit:.3f}")

    print("\n3. 学習用と評価用に分けています（8:2）...")
    X = df[FEATURE_COLUMNS]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    test_index = X_test.index
    print(f"   学習用: {len(X_train)} 件 / 評価用: {len(X_test)} 件")

    print(f"\n4. {CV_SPLITS}分割の交差検証で、モデルを比べています...")
    comparison = compare_candidates(X_train, y_train)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    best_name = comparison.iloc[0]["モデル"]
    best_model = build_candidates()[best_name]
    print(f"\n   → いちばん成績が良かったモデル: {best_name}")

    print("\n5. 選んだモデルを学習して、評価用データで確かめています...")
    best_model.fit(X_train, y_train)
    metrics = evaluate(best_model, X_test, y_test)
    test_limit = bayes_accuracy(df.loc[test_index])
    print(f"   正解率  : {metrics['accuracy']:.3f}（上限 {test_limit:.3f}）")
    print(f"   マクロF1: {metrics['macro_f1']:.3f}")
    print(f"   対数損失: {metrics['log_loss']:.3f}")
    print("   混同行列（縦：正解 / 横：予測）:")
    print(pd.DataFrame(
        metrics["confusion_matrix"], index=CATEGORIES, columns=CATEGORIES
    ).to_string())

    print("\n6. どの入力が効いているかを調べています...")
    importances = feature_importances(best_model, X_test, y_test)
    for column, value in sorted(
        importances.items(), key=lambda item: item[1]["mean"], reverse=True
    ):
        print(f"   {column:<17} {value['mean']:.3f} ± {value['std']:.3f}")

    print("\n7. 全データで学習し直して保存します...")
    final_model = build_candidates()[best_name]
    final_model.fit(X, y)
    save_model(final_model)

    card = {
        "model_name": "outing-planner-category-classifier",
        "created_at": date.today().isoformat(),
        "selected_model": best_name,
        "estimator": type(
            final_model.steps[-1][1] if isinstance(final_model, Pipeline) else final_model
        ).__name__,
        "features": FEATURE_COLUMNS,
        "classes": CATEGORIES,
        "dataset": {
            "path": DATA_PATH,
            "source": "Open-Meteo Historical Weather API (ERA5)",
            "rows": int(len(df)),
            "cities": int(raw["city"].nunique()),
            "date_from": str(raw["date"].min()),
            "date_to": str(raw["date"].max()),
            "label_counts": {key: int(value) for key, value in label_counts.items()},
            "feature_ranges": {
                column: {
                    "min": float(raw[column].min()),
                    "max": float(raw[column].max()),
                    "mean": float(raw[column].mean()),
                }
                for column in FEATURE_COLUMNS
            },
        },
        "labeling": {
            "method": "おすすめ度モデル（ルール）＋ソフトマックス抽選による弱教師あり",
            "softmax_temperature": SOFTMAX_TEMPERATURE,
            "bayes_accuracy_all": limit,
            "bayes_accuracy_test": test_limit,
        },
        "training": {
            "test_size": TEST_SIZE,
            "cv_splits": CV_SPLITS,
            "random_seed": RANDOM_SEED,
            "final_fit": "全データで学習し直し",
        },
        "cv_comparison": comparison.to_dict(orient="records"),
        "test_metrics": metrics,
        "permutation_importance": importances,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }
    save_model_card(card)

    print("\n完了！ 次は python app.py でアプリを起動してください。")


if __name__ == "__main__":
    main()
