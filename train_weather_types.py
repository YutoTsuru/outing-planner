"""
お出かけプランナー：天気タイプ分けモデルの学習スクリプト

「この日はどんなタイプの天気か」を、正解ラベルなしで自動的に分類します（k-means）。

教師なし学習には「正解」がないぶん、
    ・タイプの数をどう決めるのか
    ・その分かれ方は、たまたまではないのか
を確かめないと、ただの思いつきになってしまいます。
このスクリプトでは3つの指標で数を決め、乱数を変えても同じ分かれ方になるかを検証します。

実行方法:
    python train_weather_types.py
"""

import json
import os
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train_model
from outing_ml import data as data_module
from outing_ml.config import CONFIG, FEATURE_COLUMNS
from outing_ml.registry import ModelBundle, Registry, load_bundle, save_bundle

MODEL_NAME = "weather-type-clustering"
SEED = CONFIG.train.random_seed

DATA_PATH = CONFIG.paths.dataset
MODEL_DIR = CONFIG.paths.model_dir
MODEL_PATH = CONFIG.paths.weather_type_model
TYPES_PATH = CONFIG.paths.weather_type_card

# いくつのタイプに分けるか、試す範囲
CLUSTER_RANGE = range(3, 9)

# シルエット係数を計算するときに使う件数（全件だと時間がかかるため）
SILHOUETTE_SAMPLE = 3000

# 安定性を確かめるときに使う乱数の種
STABILITY_SEEDS = (0, 1, 2, 3, 4)


def build_model(n_clusters, seed=SEED):
    """標準化 → k-means のパイプラインを作る。

    4つの数値は単位も大きさもバラバラなので、先に標準化して
    どの項目も同じ重みで効くようにそろえます。
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)),
        ]
    )


def choose_cluster_count(X):
    """タイプの数を、3つの指標で選ぶ。

    シルエット係数だけで決めると、たまたまその指標に都合の良い数を選びかねません。
    性格のちがう3つを並べて、いちばん納得できる数を選びます。
      ・シルエット係数        大きいほど良い
      ・Calinski-Harabasz    大きいほど良い
      ・Davies-Bouldin       小さいほど良い
    """
    rows = []

    for n_clusters in CLUSTER_RANGE:
        model = build_model(n_clusters)
        labels = model.fit_predict(X)
        scaled = model[:-1].transform(X)

        rows.append(
            {
                "タイプ数": n_clusters,
                "シルエット係数": float(
                    silhouette_score(scaled, labels, sample_size=SILHOUETTE_SAMPLE,
                                     random_state=SEED)
                ),
                "CalinskiHarabasz": float(calinski_harabasz_score(scaled, labels)),
                "DaviesBouldin": float(davies_bouldin_score(scaled, labels)),
                "まとまりの悪さ": float(model["kmeans"].inertia_),
            }
        )
        print(f"   タイプ数 {n_clusters}: シルエット {rows[-1]['シルエット係数']:.3f} / "
              f"CH {rows[-1]['CalinskiHarabasz']:.0f} / "
              f"DB {rows[-1]['DaviesBouldin']:.3f}")

    return pd.DataFrame(rows)


def stability_check(X, n_clusters):
    """乱数の種を変えても、同じ分かれ方になるかを確かめる。

    k-means は最初の点の置き方で結果が変わります。
    種を変えて何度も分け、そのたびの一致度（調整ランド指数）を測ります。
    1.0 に近ければ「たまたまではない、安定した分かれ方」と言えます。
    """
    label_sets = [build_model(n_clusters, seed=seed).fit_predict(X) for seed in STABILITY_SEEDS]

    scores = []
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            scores.append(adjusted_rand_score(label_sets[i], label_sets[j]))

    return {
        "seeds": list(STABILITY_SEEDS),
        "adjusted_rand_mean": float(np.mean(scores)),
        "adjusted_rand_min": float(np.min(scores)),
    }


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


def summarize_clusters(df, X, model):
    """タイプごとに、中心の値・件数・代表的な日・おすすめカテゴリの内訳を集める。"""
    labels = model.predict(X)
    centers = model["scaler"].inverse_transform(model["kmeans"].cluster_centers_)
    scaled = model[:-1].transform(X)

    categories = None
    try:
        bundle = load_bundle(CONFIG.paths.category_model)
        categories = bundle.estimator.predict(X)
    except FileNotFoundError:
        pass

    summaries = []
    for index, center in enumerate(centers):
        members = labels == index
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
                for column, value in zip(FEATURE_COLUMNS, center, strict=True)
            },
            "examples": [f"{row.city} {row.date}" for row in nearest.itertuples()],
        }

        if categories is not None:
            counts = pd.Series(categories[members]).value_counts(normalize=True)
            summary["category_share"] = {
                key: float(round(value, 3)) for key, value in counts.items()
            }

        summaries.append(summary)

    return sorted(summaries, key=lambda item: item["count"], reverse=True)


def holdout_shares(model, path=None):
    """未来のデータでも、タイプの出方が変わっていないかを見る。

    タイプの割合が大きく変われば、気候そのものがずれてきた合図になります。
    """
    path = path or CONFIG.paths.holdout
    if not os.path.exists(path):
        return None

    frame = data_module.load_dataset(path)
    labels = model.predict(frame[FEATURE_COLUMNS])
    share = pd.Series(labels).value_counts(normalize=True).sort_index()

    return {
        "rows": int(len(frame)),
        "date_from": str(frame["date"].min()),
        "date_to": str(frame["date"].max()),
        "share": {str(key): float(round(value, 4)) for key, value in share.items()},
    }


def main():
    print("1. 気象データを読み込み・検証中...")
    df = train_model.load_dataset()
    fingerprint = data_module.dataset_fingerprint(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    print(f"   {len(df)} 行 / 指紋 {fingerprint['sha256'][:16]}...")

    print("\n2. タイプの数を決めています（3つの指標で確認）...")
    scores = choose_cluster_count(X)
    best_row = scores.loc[scores["シルエット係数"].idxmax()]
    n_clusters = int(best_row["タイプ数"])
    print(f"\n   → タイプ数は {n_clusters}（シルエット {best_row['シルエット係数']:.3f} / "
          f"DB {best_row['DaviesBouldin']:.3f}）")

    print("\n3. 乱数を変えても同じ分かれ方になるかを確認しています...")
    stability = stability_check(X, n_clusters)
    print(f"   一致度（調整ランド指数）: 平均 {stability['adjusted_rand_mean']:.3f} / "
          f"最低 {stability['adjusted_rand_min']:.3f}")
    if stability["adjusted_rand_min"] < 0.8:
        print("   ⚠ 種によって分かれ方が変わります。タイプの名前を固定して使わないでください")

    print("\n4. タイプ分けのモデルを学習中...")
    model = build_model(n_clusters)
    model.fit(X)

    print("\n5. できあがったタイプ:")
    summaries = summarize_clusters(df, X, model)
    for summary in summaries:
        center = summary["center"]
        print(f"\n   [{summary['id']}] {summary['name']}"
              f"  {summary['count']}日（{summary['share'] * 100:.1f}%）")
        print(f"        気温 {center['temperature']}℃ / 降水確率 {center['rain_probability']}% "
              f"/ 風速 {center['wind_speed']}m/s / 湿度 {center['humidity']}%")
        print(f"        代表的な日: {'、'.join(summary['examples'])}")
        if "category_share" in summary:
            share = "、".join(
                f"{key} {value * 100:.0f}%" for key, value in summary["category_share"].items()
            )
            print(f"        おすすめカテゴリの内訳: {share}")

    print("\n6. 未来のデータでのタイプの出方...")
    holdout = holdout_shares(model)
    if holdout is None:
        print("   （data/weather_jp_holdout.csv が無いので省略しました）")
    else:
        names = {str(item["id"]): item["name"] for item in summaries}
        print(f"   期間: {holdout['date_from']} 〜 {holdout['date_to']}"
              f"（{holdout['rows']} 件）")
        for type_id, value in holdout["share"].items():
            learned = next(
                (item["share"] for item in summaries if str(item["id"]) == type_id), 0.0
            )
            print(f"   [{type_id}] {names.get(type_id, ''):<28} "
                  f"学習時 {learned * 100:5.1f}% → いま {value * 100:5.1f}% "
                  f"（{(value - learned) * 100:+.1f}pt）")

    print("\n7. 保存します...")
    cluster_names = {str(item["id"]): item["name"] for item in summaries}

    card = {
        "model_name": MODEL_NAME,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "task": "天気を似たものどうしにまとめる（教師なし学習・k-means）",
        "estimator": "Pipeline(StandardScaler, KMeans)",
        "features": FEATURE_COLUMNS,
        "n_clusters": n_clusters,
        "selection": scores.to_dict(orient="records"),
        "silhouette": float(best_row["シルエット係数"]),
        "stability": stability,
        "clusters": summaries,
        "holdout": holdout,
        "data": fingerprint,
    }

    entry = Registry().record(
        model_name=MODEL_NAME,
        artifact_path=MODEL_PATH,
        task=card["task"],
        metrics={
            "n_clusters": n_clusters,
            "silhouette": card["silhouette"],
            "davies_bouldin": float(best_row["DaviesBouldin"]),
            "stability_ari_min": stability["adjusted_rand_min"],
        },
        data_fingerprint=fingerprint,
        params={"cluster_range": [CLUSTER_RANGE.start, CLUSTER_RANGE.stop - 1]},
    )

    card["version"] = entry["version"]
    card["git_sha"] = entry["git_sha"]
    card["environment"] = entry["environment"]

    save_bundle(
        MODEL_PATH,
        ModelBundle(
            estimator=model,
            feature_names=FEATURE_COLUMNS,
            model_name=MODEL_NAME,
            version=entry["version"],
            task=card["task"],
            metadata={
                "created_at": card["created_at"],
                "data": fingerprint,
                "cluster_names": cluster_names,
                "n_clusters": n_clusters,
            },
        ),
    )
    print(f"   モデルを保存しました: {MODEL_PATH}（{entry['version']}）")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(TYPES_PATH, "w", encoding="utf-8") as file:
        json.dump(card, file, ensure_ascii=False, indent=2, default=str)
    print(f"   タイプの一覧を保存しました: {TYPES_PATH}")

    print("\n完了！ 説明は doc/weather-types.md にあります。")


if __name__ == "__main__":
    main()
