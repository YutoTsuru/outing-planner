"""
お出かけプランナー：気象データ取得スクリプト

Open-Meteo の「過去の天気アーカイブ」から、日本の主要9都市の実測気象データを
ダウンロードして、1行=1日の CSV（data/weather_jp.csv）にまとめます。

このCSVが train_model.py の学習データになります。
APIキーは不要で、非商用利用は無料です（出典：Open-Meteo / ERA5 再解析データ）。

取り出す値は、アプリの入力欄と同じ4つです。
お出かけする時間帯（9時〜18時）の値だけを使って、1日ぶんに平均します。

実行方法:
    python fetch_weather.py            # data/weather_jp.csv を作る
    python fetch_weather.py --force    # すでにCSVがあっても作り直す
"""

import argparse
import os
import time

import pandas as pd
import requests

# 取得先（Open-Meteo の過去データ用エンドポイント。APIキー不要）
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# 保存先
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "weather_jp.csv")

# 取得する期間（6年ぶん）
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"

# お出かけする時間帯（9時〜18時の10時間）。この範囲の値だけを使う
OUTING_HOURS = list(range(9, 19))

# 取得する都市（北から南まで、気候がかたよらないように選んでいる）
CITIES = [
    ("札幌", 43.0621, 141.3544),
    ("仙台", 38.2682, 140.8694),
    ("新潟", 37.9161, 139.0364),
    ("東京", 35.6895, 139.6917),
    ("名古屋", 35.1815, 136.9066),
    ("大阪", 34.6937, 135.5023),
    ("高知", 33.5597, 133.5311),
    ("福岡", 33.5904, 130.4017),
    ("那覇", 26.2124, 127.6809),
]

# 「雨が降っている」とみなす1時間あたりの雨量（mm）
RAIN_THRESHOLD_MM = 0.1

# 都市ごとの取得の間隔（サーバーに負荷をかけないよう少し待つ）
REQUEST_INTERVAL_SEC = 1.0


def fetch_city_hourly(name, latitude, longitude):
    """1都市ぶんの「1時間ごとの気象データ」を取得して DataFrame で返す。"""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "wind_speed_unit": "ms",       # 風速の単位を m/s にそろえる（既定は km/h）
        "timezone": "Asia/Tokyo",      # 日本時間で日付を区切る
    }

    response = requests.get(ARCHIVE_URL, params=params, timeout=120)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(hourly["time"]),
            "temperature": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "wind_speed": hourly["wind_speed_10m"],
            "precipitation": hourly["precipitation"],
        }
    )
    df["city"] = name
    return df


def summarize_by_day(df):
    """1時間ごとのデータを、1日1行のデータにまとめる。"""
    # お出かけ時間帯だけを残す
    df = df[df["time"].dt.hour.isin(OUTING_HOURS)].copy()

    # 欠測（値が無い時間）がある日は、平均がゆがむので後でまとめて落とす
    df["date"] = df["time"].dt.date
    df["is_rainy_hour"] = (df["precipitation"] >= RAIN_THRESHOLD_MM).astype(float)

    grouped = df.groupby(["city", "date"]).agg(
        temperature=("temperature", "mean"),
        humidity=("humidity", "mean"),
        wind_speed=("wind_speed", "mean"),
        rain_hours=("is_rainy_hour", "sum"),
        hours=("time", "count"),
        missing=("temperature", lambda values: values.isna().sum()),
    )

    # 10時間そろっていない日・欠測がある日は使わない
    grouped = grouped[(grouped["hours"] == len(OUTING_HOURS)) & (grouped["missing"] == 0)]

    # 「10時間のうち何時間 雨だったか」を割合（%）にして、降水確率のかわりに使う
    grouped["rain_probability"] = grouped["rain_hours"] / len(OUTING_HOURS) * 100

    result = grouped.reset_index()[
        ["city", "date", "temperature", "rain_probability", "wind_speed", "humidity"]
    ]
    result["temperature"] = result["temperature"].round(1)
    result["rain_probability"] = result["rain_probability"].round(0).astype(int)
    result["wind_speed"] = result["wind_speed"].round(1)
    result["humidity"] = result["humidity"].round(0).astype(int)
    return result


def download_all():
    """全都市ぶんを取得して、1つの DataFrame にまとめる。"""
    frames = []

    for index, (name, latitude, longitude) in enumerate(CITIES, start=1):
        print(f"  [{index}/{len(CITIES)}] {name} を取得中...", flush=True)
        hourly = fetch_city_hourly(name, latitude, longitude)
        daily = summarize_by_day(hourly)
        print(f"      {len(daily)} 日ぶん")
        frames.append(daily)

        if index < len(CITIES):
            time.sleep(REQUEST_INTERVAL_SEC)

    return pd.concat(frames, ignore_index=True).sort_values(["city", "date"])


def main():
    parser = argparse.ArgumentParser(description="気象データをダウンロードしてCSVに保存する")
    parser.add_argument("--force", action="store_true", help="CSVがすでにあっても作り直す")
    args = parser.parse_args()

    if os.path.exists(DATA_PATH) and not args.force:
        print(f"すでにデータがあります: {DATA_PATH}")
        print("作り直すときは python fetch_weather.py --force を実行してください。")
        return

    print(f"Open-Meteo から気象データを取得します（{START_DATE} 〜 {END_DATE}）")
    df = download_all()

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    print(f"\n保存しました: {DATA_PATH}（{len(df)} 行）")
    print("次は python train_model.py でモデルを学習してください。")


if __name__ == "__main__":
    main()
