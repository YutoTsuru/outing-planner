"""都市どうしのあしたの予報を比べる（キャッシュつき）。

47都市ぶんの予報を作るには、都市ごとに外部（Open-Meteo）へ通信します。
毎回のページ表示でこれをやり直すと、遅いうえに取得元の利用上限に
当たりやすくなります。そこで結果をしばらく覚えておき（メモリ内キャッシュ）、
一定時間が経つまでは使い回します。

このキャッシュはプロセス内だけのものです。gunicorn を複数ワーカーで
動かすと、ワーカーごとに別々のキャッシュを持ちます（同じ内容を
ワーカーの数だけ取得し直すことになりますが、動作としては問題ありません）。
"""

import time

from outing_ml.forecasting import ForecastService

# キャッシュの有効時間（秒）。あしたの天気は数十分単位では変わらないので、
# 短くしすぎる必要はない。一方で「学習し直した」等の変化には追随したいので、
# 数十分〜1時間程度にとどめる。
CACHE_TTL_SECONDS = 1800

# キャッシュの中身。{"at": 作った時刻, "data": 比較結果}
_cache: dict = {"at": 0.0, "data": None}


def _rank(comparison: dict) -> list[dict]:
    """日和度が高い順に並べ、画面・APIで使いやすい形にする。"""
    rows = []
    for forecast in comparison["results"]:
        recommendation = forecast.recommendation
        rows.append(
            {
                "city": forecast.city,
                "target_date": forecast.target_date,
                "weather": forecast.weather,
                "category": recommendation.category if recommendation else None,
                "comfort_score": recommendation.comfort_score if recommendation else None,
                "weather_type_name": (
                    recommendation.weather_type_name if recommendation else None
                ),
                "confidence": recommendation.confidence if recommendation else None,
            }
        )

    # 日和度が無いものは最後に回す（None は比較できないため -1 扱い）
    rows.sort(key=lambda row: row["comfort_score"] if row["comfort_score"] is not None else -1,
             reverse=True)
    return rows


def get_comparison(service: ForecastService = None, force_refresh: bool = False) -> dict:
    """都市比較の結果を返す。有効なキャッシュがあればそれを使う。"""
    now = time.time()
    age = now - _cache["at"]

    if not force_refresh and _cache["data"] is not None and age < CACHE_TTL_SECONDS:
        return {**_cache["data"], "cache_age_seconds": round(age)}

    service = service or ForecastService.load()
    comparison = service.compare_tomorrow()

    data = {
        "rankings": _rank(comparison),
        "errors": comparison["errors"],
        "fetched_at": now,
        "ttl_seconds": CACHE_TTL_SECONDS,
    }
    _cache["at"] = now
    _cache["data"] = data
    return {**data, "cache_age_seconds": 0}


def clear_cache() -> None:
    """キャッシュを空にする（テスト・強制更新用）。"""
    _cache["at"] = 0.0
    _cache["data"] = None
