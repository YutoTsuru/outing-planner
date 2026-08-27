"""
お出かけプランナー：天気タイプ分けモデルの学習スクリプト

「この日はどんなタイプの天気か」を、正解ラベルなしで自動的に分類します。
カテゴリ予測（train_model.py）が正解を教えてもらう「教師あり学習」なのに対して、
こちらは正解を使わない「教師なし学習」（k-means クラスタリング）です。

    例：「さわやかで過ごしやすい日」「蒸し暑い日」「冷たい雨の日」…

同じ outdoor でも「さわやかな日」と「暑いけれど晴れの日」では
おすすめしたい場所が変わります。その差をつけるための材料になります。

実行方法:
    python train_weather_types.py
"""

import json
import os
import platform
from datetime import date

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train_model

# 乱数の種
RANDOM_SEED = 42

# 読み込み先・保存先
DATA_PATH = train_model.DATA_PATH
MODEL_DIR = train_model.MODEL_DIR
MODEL_PATH = os.path.join(MODEL_DIR, "weather_type_model.pkl")
TYPES_PATH = os.path.join(MODEL_DIR, "weather_types.json")

# 入力に使う列（ほかのモデルと同じ4つ）
FEATURE_COLUMNS = train_model.FEATURE_COLUMNS

# いくつのタイプに分けるか、試す範囲
CLUSTER_RANGE = range(3, 9)

# シルエット係数を計算するときに使う件数（全件だと時間がかかるため）
SILHOUETTE_SAMPLE = 3000


# ---------------------------------------------------------------
# 1. タイプ分けのモデル
# ---------------------------------------------------------------

def build_model(n_clusters):
    """k-means のモデルを作る。

    4つの数値は単位も大きさもバラバラ（気温は数十、湿度は百）なので、
    先に標準化して、どの項目も同じ重みで効くようにそろえます。
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_SEED),
            ),
        ]
    )


def choose_cluster_count(X):
    """タイプの数をいくつにするか、シルエット係数で選ぶ。

    シルエット係数は「まとまりの良さ」を -1〜1 で表す指標で、
    大きいほど、それぞれのタイプがはっきり分かれています。
    """
    rows = []

    for n_clusters in CLUSTER_RANGE:
        model = build_model(n_clusters)
        labels = model.fit_predict(X)
        score = silhouette_score(
            model[:-1].transform(X),
            labels,
            sample_size=SILHOUETTE_SAMPLE,
            random_state=RANDOM_SEED,
        )
        rows.append(
            {
                "タイプ数": n_clusters,
                "シルエット係数": float(score),
                "まとまりの悪さ": float(model["kmeans"].inertia_),
            }
        )
        print(f"   タイプ数 {n_clusters}: シルエット係数 {score:.3f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# 2. タイプに名前をつける
# ---------------------------------------------------------------

def describe_center(center):
    """タイプの中心の値から、日本語の名前を組み立てる。"""
    temperature, rain, wind, humidity = center

    if temperature < 8:
        heat = "寒い"
    elif temperature < 16:
        heat = "肌寒い"
    elif temperature < 26:
        heat = "過ごしやすい"
    elif temperature < 29:
        heat = "暑い"
    else:
        heat = "真夏日の"

    if rain >= 60:
        sky = "雨"
    elif rain >= 25:
        sky = "降ったりやんだり"
    elif rain >= 12:
        sky = "くもりがち"
    else:
        sky = "晴れ"

    extras = []
    if humidity >= 78:
        extras.append("じめじめ")
    elif humidity <= 55:
        extras.append("からっと")
    if wind >= 6:
        extras.append("風が強い")

    name = f"{heat}{sky}の日"
    if extras:
        name = f"{heat}{sky}の日（{'・'.join(extras)}）"
    return name


# ---------------------------------------------------------------
# 3. タイプごとのまとめ
# ---------------------------------------------------------------

def summarize_clusters(df, X, model):
    """タイプごとに、中心の値・件数・代表的な日・おすすめカテゴリの内訳を集める。"""
    labels = model.predict(X)
    centers = model["scaler"].inverse_transform(model["kmeans"].cluster_centers_)
    scaled = model[:-1].transform(X)

    # 参考として、カテゴリ予測モデルの答えも見てみる（あれば）
    categories = None
    if os.path.exists(train_model.MODEL_PATH):
        category_model = joblib.load(train_model.MODEL_PATH)
        categories = category_model.predict(X)

    summaries = []
    for index, center in enumerate(centers):
        members = labels == index
        # 中心にいちばん近い日を、そのタイプの代表として3つ選ぶ
        distances = np.linalg.norm(
            scaled[members] - model["kmeans"].cluster_centers_[index], axis=1
        )
        nearest = df[members].iloc[np.argsort(distances)[:3]]

        summary = {
            "id": int(index),
            "name": describe_center(center),
            "count": int(members.sum()),
            "share": float(members.mean()),
            "center": {
                column: float(round(value, 1))
                for column, value in zip(FEATURE_COLUMNS, center)
            },
            "examples": [
                f"{row.city} {row.date}" for row in nearest.itertuples()
            ],
        }

        if categories is not None:
            counts = pd.Series(categories[members]).value_counts(normalize=True)
            summary["category_share"] = {
                key: float(round(value, 3)) for key, value in counts.items()
            }

        summaries.append(summary)

    return sorted(summaries, key=lambda item: item["count"], reverse=True)


# ---------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------

def main():
    print("1. 気象データを読み込み中...")
    df = train_model.load_dataset()
    X = df[FEATURE_COLUMNS]
    print(f"   データ件数: {len(df)}")

    print("\n2. タイプの数を決めています（シルエット係数がいちばん大きいものを選ぶ）...")
    scores = choose_cluster_count(X)
    best_row = scores.loc[scores["シルエット係数"].idxmax()]
    n_clusters = int(best_row["タイプ数"])
    print(f"\n   → タイプ数は {n_clusters} に決めました"
          f"（シルエット係数 {best_row['シルエット係数']:.3f}）")

    print("\n3. タイプ分けのモデルを学習中...")
    model = build_model(n_clusters)
    model.fit(X)

    print("\n4. できあがったタイプ:")
    summaries = summarize_clusters(df, X, model)
    for summary in summaries:
        center = summary["center"]
        print(f"\n   [{summary['id']}] {summary['name']}"
              f"  {summary['count']}日（{summary['share'] * 100:.1f}%）")
        print(f"        気温 {center['temperature']}℃ ／ 降水確率 {center['rain_probability']}% "
              f"／ 風速 {center['wind_speed']}m/s ／ 湿度 {center['humidity']}%")
        print(f"        代表的な日: {'、'.join(summary['examples'])}")
        if "category_share" in summary:
            share = "、".join(
                f"{key} {value * 100:.0f}%" for key, value in summary["category_share"].items()
            )
            print(f"        おすすめカテゴリの内訳: {share}")

    print("\n5. 保存します...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"モデルを保存しました: {MODEL_PATH}")

    card = {
        "model_name": "outing-planner-weather-types",
        "created_at": date.today().isoformat(),
        "task": "天気を似たものどうしにまとめる（教師なし学習・k-means）",
        "estimator": "Pipeline(StandardScaler, KMeans)",
        "features": FEATURE_COLUMNS,
        "n_clusters": n_clusters,
        "selection": scores.to_dict(orient="records"),
        "silhouette": float(best_row["シルエット係数"]),
        "clusters": summaries,
        "dataset": {"path": DATA_PATH, "rows": int(len(df))},
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    with open(TYPES_PATH, "w", encoding="utf-8") as file:
        json.dump(card, file, ensure_ascii=False, indent=2)
    print(f"タイプの一覧を保存しました: {TYPES_PATH}")

    print("\n完了！ 説明は doc/weather-types.md にあります。")


if __name__ == "__main__":
    main()
