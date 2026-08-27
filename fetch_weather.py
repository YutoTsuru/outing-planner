"""
お出かけプランナー：気象データ取得スクリプト

Open-Meteo の「過去の天気アーカイブ」から、日本の主要9都市の実測気象データを
ダウンロードして、1行=1日の CSV にまとめます。

    data/weather_jp.csv          学習・検証に使う（2019〜2024年）
    data/weather_jp_holdout.csv  学習後に一度だけ使う「未来のデータ」（2025年〜）

学習データと未来データを分けているのは、
「学習が終わったあとに一度だけ見る、まったく触っていないデータ」を残しておくためです。
何度も見て調整したデータでは、成績が良く見えて当たり前になってしまいます。

APIキーは不要で、非商用利用は無料です（出典：Open-Meteo / ECMWF ERA5、CC BY 4.0）。

実行方法:
    python fetch_weather.py              # 学習データを作る
    python fetch_weather.py --holdout    # 未来データを作る
    python fetch_weather.py --all        # 両方
    python fetch_weather.py --force      # すでにあっても作り直す
"""

import argparse
import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

from outing_ml.config import CONFIG
from outing_ml.data import validate_frame

# 取得先（Open-Meteo の過去データ用エンドポイント。APIキー不要）
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DATA_DIR = CONFIG.paths.data_dir
DATA_PATH = CONFIG.paths.dataset
HOLDOUT_PATH = CONFIG.paths.holdout


def fetch_city_hourly(name, latitude, longitude, start_date, end_date):
    """1都市ぶんの「1時間ごとの気象データ」を取得して DataFrame で返す。"""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "wind_speed_unit": "ms",       # 風速の単位を m/s にそろえる（既定は km/h）
        "timezone": "Asia/Tokyo",      # 日本時間で日付を区切る
    }

    response = requests.get(ARCHIVE_URL, params=params,
                            timeout=CONFIG.data.request_timeout_sec)
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
    hours = list(CONFIG.data.outing_hours)
    df = df[df["time"].dt.hour.isin(hours)].copy()

    df["date"] = df["time"].dt.date
    df["is_rainy_hour"] = (
        df["precipitation"] >= CONFIG.data.rain_threshold_mm
    ).astype(float)

    grouped = df.groupby(["city", "date"]).agg(
        temperature=("temperature", "mean"),
        humidity=("humidity", "mean"),
        wind_speed=("wind_speed", "mean"),
        rain_hours=("is_rainy_hour", "sum"),
        hours=("time", "count"),
        missing=("temperature", lambda values: values.isna().sum()),
    )

    # 10時間そろっていない日・欠測がある日は使わない
    grouped = grouped[(grouped["hours"] == len(hours)) & (grouped["missing"] == 0)]

    # 「10時間のうち何時間 雨だったか」を割合（%）にして、降水確率のかわりに使う
    grouped["rain_probability"] = grouped["rain_hours"] / len(hours) * 100

    result = grouped.reset_index()[
        ["city", "date", "temperature", "rain_probability", "wind_speed", "humidity"]
    ]
    result["temperature"] = result["temperature"].round(1)
    result["rain_probability"] = result["rain_probability"].round(0).astype(int)
    result["wind_speed"] = result["wind_speed"].round(1)
    result["humidity"] = result["humidity"].round(0).astype(int)
    return result


def download_range(start_date, end_date):
    """全都市ぶんを取得して、1つの DataFrame にまとめる。"""
    frames = []
    cities = CONFIG.data.cities

    for index, (name, latitude, longitude) in enumerate(cities, start=1):
        print(f"  [{index}/{len(cities)}] {name} を取得中...", flush=True)
        hourly = fetch_city_hourly(name, latitude, longitude, start_date, end_date)
        daily = summarize_by_day(hourly)
        print(f"      {len(daily)} 日ぶん")
        frames.append(daily)

        if index < len(cities):
            time.sleep(CONFIG.data.request_interval_sec)

    return pd.concat(frames, ignore_index=True).sort_values(["city", "date"])


def download_all():
    """学習データの期間を取得する（train_model.py から呼ばれる）。"""
    return download_range(CONFIG.data.start_date, CONFIG.data.end_date)


def holdout_end_date():
    """未来データの終わりの日。

    アーカイブは数日おくれて更新されるため、直近1週間は取りません。
    """
    return (date.today() - timedelta(days=CONFIG.data.holdout_lag_days)).isoformat()


def save(df, path):
    """検証してから保存する。"""
    report = validate_frame(df)
    if not report.ok:
        print("  検証に失敗しました:")
        for error in report.errors:
            print(f"    - {error}")
        raise SystemExit(1)

    for warning in report.warnings:
        print(f"  ⚠ {warning}")

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n保存しました: {path}（{len(df)} 行）")


def build(path, start_date, end_date, label, force):
    """1つのデータセットを作る。"""
    if os.path.exists(path) and not force:
        print(f"すでにあります: {path}（作り直すときは --force）")
        return

    print(f"\n{label}を取得します（{start_date} 〜 {end_date}）")
    save(download_range(start_date, end_date), path)


def main():
    parser = argparse.ArgumentParser(description="気象データをダウンロードしてCSVに保存する")
    parser.add_argument("--holdout", action="store_true", help="未来データだけを作る")
    parser.add_argument("--all", action="store_true", help="学習データと未来データの両方を作る")
    parser.add_argument("--force", action="store_true", help="すでにあっても作り直す")
    args = parser.parse_args()

    want_train = args.all or not args.holdout
    want_holdout = args.all or args.holdout

    if want_train:
        build(DATA_PATH, CONFIG.data.start_date, CONFIG.data.end_date,
              "学習データ", args.force)

    if want_holdout:
        build(HOLDOUT_PATH, CONFIG.data.holdout_start_date, holdout_end_date(),
              "未来データ（ホールドアウト）", args.force)

    print("\n次は python train_all.py でモデルを学習してください。")


if __name__ == "__main__":
    main()
