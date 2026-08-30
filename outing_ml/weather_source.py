"""気象データの取得元（Open-Meteo）。

学習データを作る fetch_weather.py と、翌日予報を返す API の両方がここを使います。
同じ前処理（お出かけ時間帯の平均・雨だった時間の割合）を2か所に書くと、
学習したときと本番で特徴量の作り方がズレて、静かに精度が落ちるためです。

出典: Open-Meteo Historical Weather API（ECMWF ERA5）, CC BY 4.0
"""

import time
from datetime import date, timedelta

import pandas as pd
import requests

from outing_ml.config import CONFIG

# 過去データ用エンドポイント（APIキー不要）
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# 取得する1時間ごとの項目
HOURLY_VARIABLES = "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"


class UnknownCityError(ValueError):
    """学習に使っていない都市を指定されたときに投げる例外。"""


def city_table() -> dict[str, tuple[float, float]]:
    """都市名 → (緯度, 経度)。"""
    return {name: (latitude, longitude) for name, latitude, longitude in CONFIG.data.cities}


def city_names() -> list[str]:
    """使える都市の一覧。"""
    return [name for name, _, _ in CONFIG.data.cities]


def resolve_city(name: str) -> tuple[str, float, float]:
    """都市名から緯度経度を引く。

    学習データに入っていない都市は受け付けません。モデルは都市名を入力に使っており、
    知らない都市を渡すと当てずっぽうの予測を返してしまうためです。
    """
    table = city_table()
    if name not in table:
        raise UnknownCityError(
            f"{name} は使えません。使えるのは次の9都市です: {'、'.join(city_names())}"
        )
    latitude, longitude = table[name]
    return name, latitude, longitude


class RateLimitedError(RuntimeError):
    """取得元の利用上限に当たったときに投げる例外。"""


def _request_hourly(latitude: float, longitude: float,
                    start_date: str, end_date: str) -> dict:
    """1回ぶんの取得。上限に当たったら RateLimitedError にする。"""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": HOURLY_VARIABLES,
        "wind_speed_unit": "ms",       # 風速の単位を m/s にそろえる（既定は km/h）
        "timezone": "Asia/Tokyo",      # 日本時間で日付を区切る
    }

    response = requests.get(ARCHIVE_URL, params=params,
                            timeout=CONFIG.data.request_timeout_sec)

    if response.status_code == 429:
        # サーバが待ち時間を教えてくれることがある
        wait = response.headers.get("Retry-After")
        raise RateLimitedError(wait or "")

    response.raise_for_status()
    return response.json()["hourly"]


def fetch_city_hourly(name: str, latitude: float, longitude: float,
                      start_date: str, end_date: str,
                      max_retries: int = 4) -> pd.DataFrame:
    """1都市ぶんの「1時間ごとの気象データ」を取得する。

    取得元（Open-Meteo）の無料枠は、リクエストの回数ではなく
    「変数の数 × 日数」で消費されます。長い期間を一度に頼むと上限に当たるため、
    **1年ずつに切って**取り、429 が返ったら待ってからやり直します。
    """
    frames = []
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    for year in range(start_year, end_year + 1):
        chunk_start = max(start_date, f"{year}-01-01")
        chunk_end = min(end_date, f"{year}-12-31")

        for attempt in range(max_retries):
            try:
                hourly = _request_hourly(latitude, longitude, chunk_start, chunk_end)
                break
            except RateLimitedError as error:
                if attempt == max_retries - 1:
                    raise
                # Retry-After があればそれに従い、無ければ 30秒 → 60 → 120 と待つ
                wait = int(str(error)) if str(error).isdigit() else 30 * (2 ** attempt)
                print(f"      利用上限のため {wait} 秒待ちます（{name} {year}年）", flush=True)
                time.sleep(wait)
        else:  # pragma: no cover - 上の break / raise で必ず抜ける
            raise RateLimitedError("")

        frames.append(
            pd.DataFrame(
                {
                    "time": pd.to_datetime(hourly["time"]),
                    "temperature": hourly["temperature_2m"],
                    "humidity": hourly["relative_humidity_2m"],
                    "wind_speed": hourly["wind_speed_10m"],
                    "precipitation": hourly["precipitation"],
                }
            )
        )
        time.sleep(CONFIG.data.request_interval_sec)

    frame = pd.concat(frames, ignore_index=True)
    frame["city"] = name
    return frame


def summarize_by_day(frame: pd.DataFrame) -> pd.DataFrame:
    """1時間ごとのデータを、1日1行にまとめる。

    お出かけする時間帯（9〜18時の10時間）だけを使い、
    降水確率は「そのうち何時間 雨だったか」の割合で表します。
    """
    hours = list(CONFIG.data.outing_hours)
    frame = frame[frame["time"].dt.hour.isin(hours)].copy()

    frame["date"] = frame["time"].dt.date
    frame["is_rainy_hour"] = (
        frame["precipitation"] >= CONFIG.data.rain_threshold_mm
    ).astype(float)

    grouped = frame.groupby(["city", "date"]).agg(
        temperature=("temperature", "mean"),
        humidity=("humidity", "mean"),
        wind_speed=("wind_speed", "mean"),
        rain_hours=("is_rainy_hour", "sum"),
        hours=("time", "count"),
        missing=("temperature", lambda values: values.isna().sum()),
    )

    # 10時間そろっていない日・欠測がある日は使わない
    grouped = grouped[(grouped["hours"] == len(hours)) & (grouped["missing"] == 0)]
    grouped["rain_probability"] = grouped["rain_hours"] / len(hours) * 100

    result = grouped.reset_index()[
        ["city", "date", "temperature", "rain_probability", "wind_speed", "humidity"]
    ]
    result["temperature"] = result["temperature"].round(1)
    result["rain_probability"] = result["rain_probability"].round(0).astype(int)
    result["wind_speed"] = result["wind_speed"].round(1)
    result["humidity"] = result["humidity"].round(0).astype(int)
    return result


def download_range(start_date: str, end_date: str, cities=None,
                   progress=None, on_city=None, done_cities=()) -> pd.DataFrame:
    """指定した期間ぶんを、都市ごとに取得する。

    途中で利用上限に当たったら、そこまでに取れたぶんを返します（例外にしません）。
    残りは日を改めて --resume で続きから取れます。
    """
    cities = cities or CONFIG.data.cities
    remaining = [city for city in cities if city[0] not in set(done_cities)]
    frames = []

    for index, (name, latitude, longitude) in enumerate(remaining, start=1):
        if progress:
            progress(index, len(remaining), name)

        try:
            daily = summarize_by_day(
                fetch_city_hourly(name, latitude, longitude, start_date, end_date)
            )
        except RateLimitedError:
            print(f"      取得元の利用上限に達しました。{name} 以降は次回に回します", flush=True)
            break

        frames.append(daily)
        if on_city:
            on_city(name, daily)

    if not frames:
        return pd.DataFrame(
            columns=["city", "date", "temperature", "rain_probability",
                     "wind_speed", "humidity"]
        )
    return pd.concat(frames, ignore_index=True).sort_values(["city", "date"])


def recent_daily(city: str, days: int = 10, end_date: str | None = None) -> pd.DataFrame:
    """1都市の「直近の実測」を取ってくる（翌日予報の材料）。

    実測アーカイブは数日おくれて公開されます。ここでは今日までを要求し、
    まだ公開されていない日は summarize_by_day が落とします
    （10時間そろっていない日は使わない、という同じ規則で弾かれる）。
    こうすると「取得できる中でいちばん新しい日」から予測できます。
    """
    name, latitude, longitude = resolve_city(city)

    if end_date is None:
        end_date = date.today().isoformat()
    start_date = (date.fromisoformat(end_date) - timedelta(days=days)).isoformat()

    hourly = fetch_city_hourly(name, latitude, longitude, start_date, end_date)
    return summarize_by_day(hourly).sort_values("date").reset_index(drop=True)
