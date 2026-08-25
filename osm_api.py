"""お出かけプランナー：OpenStreetMap から実在するスポットを探す部分

Google Maps API はキーと請求先アカウントの設定が必要ですが、
OpenStreetMap（OSM）の Nominatim は無料・キーなしで使えます。
このファイルでは Nominatim を使って、次の3つを行います。

1. 地名 → 緯度経度      （例：「京都市」→ 35.01, 135.76）
2. 緯度経度 → 地名      （現在地の表示に使う）
3. まわりの施設の検索    （例：「神社」→ 八坂神社、下鴨神社 …）

Google Maps が使えないときは、こちらから実在の施設名を取ってきます。

※ Nominatim は無料で公開されている共有サーバーです。利用ルールとして
   「1秒に1回まで」「User-Agent を名乗る」ことが決められているので、
   このファイルではリクエストの間隔をあけ、同じ検索は覚えておく（キャッシュ）
   ようにしています。
※ 取得したデータは © OpenStreetMap contributors（ODbL）です。
"""

import math
import time
import urllib.parse

import requests

# ---------------------------------------------------------------
# 設定
# ---------------------------------------------------------------

SEARCH_URL = "https://nominatim.openstreetmap.org/search"
REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"

# 利用ルールとして、アプリ名を名乗る必要がある
HEADERS = {"User-Agent": "outing-planner/1.0 (school project)"}

# 応答を待つ最大秒数
TIMEOUT_SECONDS = 15

# リクエストの最低間隔（秒）。1秒に1回までのルールを守るため
MIN_INTERVAL_SECONDS = 1.1

# 同じ検索の結果を覚えておく場所（アプリを再起動すると消える）
_cache = {}

# 最後にリクエストした時刻
_last_request_time = 0.0


class OsmError(Exception):
    """OpenStreetMap の呼び出しに失敗したときのエラー。"""


def request_json(url, params):
    """Nominatim に問い合わせて、JSONを返す（間隔をあけて呼ぶ）。"""
    global _last_request_time

    waiting = MIN_INTERVAL_SECONDS - (time.time() - _last_request_time)
    if waiting > 0:
        time.sleep(waiting)

    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT_SECONDS)
        data = response.json()
    except (requests.RequestException, ValueError) as error:
        raise OsmError(f"OpenStreetMap に接続できませんでした（{error}）")
    finally:
        _last_request_time = time.time()

    return data


# ---------------------------------------------------------------
# 地名 ⇔ 緯度経度
# ---------------------------------------------------------------

def geocode(address):
    """地名から緯度経度を調べる。

    戻り値: (緯度, 経度, 地名) のタプル
    """
    results = request_json(
        SEARCH_URL,
        {
            "q": address,
            "format": "json",
            "limit": 1,
            "accept-language": "ja",
            "countrycodes": "jp",
        },
    )

    if not results:
        raise OsmError("その地名は見つかりませんでした")

    top = results[0]
    # 「京都市, 京都府, 日本」→「京都府京都市」のように短くする
    parts = [part.strip() for part in top.get("display_name", "").split(",")]
    short_name = "".join(reversed(parts[:2])) if len(parts) >= 2 else address
    return float(top["lat"]), float(top["lon"]), short_name or address


def reverse_geocode(latitude, longitude):
    """緯度経度から地名を調べる。失敗したときは None を返す。"""
    try:
        data = request_json(
            REVERSE_URL,
            {
                "lat": latitude,
                "lon": longitude,
                "format": "json",
                "zoom": 12,          # 市区町村くらいの細かさ
                "accept-language": "ja",
            },
        )
    except OsmError:
        return None

    address = data.get("address", {})
    # 「京都府京都市」のように、都道府県＋市区町村で組み立てる
    parts = [
        address.get("province") or address.get("state"),
        address.get("city") or address.get("town") or address.get("village")
        or address.get("county"),
    ]
    name = "".join(part for part in parts if part)
    return name or data.get("display_name")


# ---------------------------------------------------------------
# まわりの施設をさがす
# ---------------------------------------------------------------

def distance_m(lat1, lon1, lat2, lon2):
    """2地点のだいたいの距離（メートル）を計算する。"""
    earth_radius = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return int(2 * earth_radius * math.asin(math.sqrt(a)))


def view_box(latitude, longitude, radius_m):
    """緯度経度のまわり「radius_m メートル四方」の範囲を作る。"""
    delta_lat = radius_m / 111000
    delta_lon = radius_m / (111000 * max(math.cos(math.radians(latitude)), 0.1))
    return (
        f"{longitude - delta_lon},{latitude + delta_lat},"
        f"{longitude + delta_lon},{latitude - delta_lat}"
    )


def search_spots(keyword, latitude, longitude, radius_m=3000, limit=6):
    """指定した場所のまわりで、キーワードに合う施設を近い順に探す。

    「公園」のような種類でも、「はま寿司」のようなお店の名前でも検索できる。
    """
    cache_key = (keyword, round(latitude, 3), round(longitude, 3), int(radius_m), limit)
    if cache_key in _cache:
        return _cache[cache_key]

    results = request_json(
        SEARCH_URL,
        {
            "q": keyword,
            "viewbox": view_box(latitude, longitude, radius_m),
            "bounded": 1,          # この範囲の中だけから探す
            "format": "json",
            "limit": limit,
            "accept-language": "ja",
            "countrycodes": "jp",
        },
    )

    spots = []
    for item in results:
        display = item.get("display_name", "")
        name = item.get("name") or display.split(",")[0]
        if not name:
            continue

        spot_lat, spot_lon = float(item["lat"]), float(item["lon"])
        spots.append(
            {
                "name": name,
                "address": display,
                "rating": None,
                "rating_count": 0,
                "distance_m": distance_m(latitude, longitude, spot_lat, spot_lon),
                "url": map_link(name, spot_lat, spot_lon),
                "keyword": keyword,
                "source": "osm",
            }
        )

    spots.sort(key=lambda spot: spot["distance_m"])
    _cache[cache_key] = spots
    return spots


def map_link(name, latitude, longitude):
    """施設名でGoogleマップを開くリンクを作る（APIキー不要）。"""
    quoted = urllib.parse.quote(name)
    return f"https://www.google.com/maps/search/{quoted}/@{latitude},{longitude},17z"
