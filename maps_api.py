"""お出かけプランナー：Google Maps API との連携部分

このファイルでは、次の2つを担当します。

1. 地名（市区町村・都道府県など）から緯度経度を調べる（Geocoding API）
2. 緯度経度のまわりのスポットを検索する（Places API (New) の Text Search）

APIキーを設定すると実際のAPIを呼びます。設定方法は次の2つのどちらかです。

  A) ファイルに書く（かんたん・おすすめ）
     このフォルダに google_maps_api_key.txt を作り、キーだけを書いて保存する
     （このファイルは .gitignore に入れてあるので、GitHubには上がりません）

  B) 環境変数に入れる
     Windows (PowerShell)  $env:GOOGLE_MAPS_API_KEY = "取得したキー"
     macOS / Linux         export GOOGLE_MAPS_API_KEY="取得したキー"

APIキーが無いときは「オフラインモード」になり、APIを呼ばずに
Googleマップの検索リンクだけを作ります（授業中でも動くようにするため）。
"""

import os
import urllib.parse

import requests

# ---------------------------------------------------------------
# 設定
# ---------------------------------------------------------------

# APIキーを書いておくファイルの名前（環境変数が無いときはこちらを読む）
API_KEY_FILE = "google_maps_api_key.txt"


def read_api_key():
    """APIキーを読み込む。環境変数 → ファイル の順に探す。

    どちらにも無いときは空文字を返す（＝オフラインモードで動く）。
    """
    key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if key:
        return key

    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, encoding="utf-8") as file:
            # コメント行（#で始まる行）と空行は読み飛ばす
            for line in file:
                line = line.strip()
                if line and not line.startswith("#"):
                    return line

    return ""


# APIキー（無ければ空文字。空のときはオフラインモードで動く）
API_KEY = read_api_key()

# 使用するAPIのURL
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"

# Places API (New) で受け取りたい項目（FieldMask）
PLACES_FIELDS = ",".join(
    [
        "places.displayName",
        "places.formattedAddress",
        "places.rating",
        "places.userRatingCount",
        "places.location",
        "places.googleMapsUri",
    ]
)

# APIの応答を待つ最大秒数
TIMEOUT_SECONDS = 10


def has_api_key():
    """APIキーが設定されているかどうかを返す。"""
    return bool(API_KEY)


class MapsError(Exception):
    """Google Maps API の呼び出しに失敗したときのエラー。"""


# 一度「利用を拒否」されたら理由を覚えておき、以降はAPIを呼ばない
# （設定を直したあとは、アプリを再起動すればまた呼ぶようになります）
_denied_reason = None


def explain_denied(raw_message):
    """Googleからの英語のエラーを、分かりやすい日本語に言いかえる。"""
    text = (raw_message or "").lower()

    if "billing" in text:
        return "Google Cloud の請求先アカウントが未設定のため、Google Maps を利用できません"
    if "not enabled" in text or "has not been used" in text:
        return "Places API (New) / Geocoding API が有効になっていません"
    if "permission" in text or "denied" in text or "unregistered" in text:
        return (
            "Google Maps の利用が拒否されました"
            "（請求先アカウントの未設定、またはAPIキーの制限が原因のことが多いです）"
        )
    if "quota" in text or "resource_exhausted" in text:
        return "Google Maps の利用上限に達しました"

    return f"Google Maps を利用できませんでした（{raw_message}）"


def remember_denied(raw_message):
    """拒否された理由を覚えて、日本語にしたエラーを返す。"""
    global _denied_reason
    _denied_reason = explain_denied(raw_message)
    return MapsError(_denied_reason)


def denied_reason():
    """すでに拒否されているなら、その理由を返す（まだなら None）。"""
    return _denied_reason


# ---------------------------------------------------------------
# 地名 ⇔ 緯度経度
# ---------------------------------------------------------------

def geocode(address):
    """地名（例：「京都市」「神奈川県横浜市」）から緯度経度を調べる。

    戻り値: (緯度, 経度, 正式な地名) のタプル
    """
    if not has_api_key():
        raise MapsError("APIキーが設定されていません。")
    if _denied_reason:
        raise MapsError(_denied_reason)

    params = {"address": address, "language": "ja", "region": "jp", "key": API_KEY}
    try:
        response = requests.get(GEOCODE_URL, params=params, timeout=TIMEOUT_SECONDS)
        data = response.json()
    except requests.RequestException as error:
        raise MapsError(f"通信に失敗しました: {error}")

    if data.get("status") in ("REQUEST_DENIED", "OVER_QUERY_LIMIT"):
        raise remember_denied(data.get("error_message") or data.get("status"))

    if data.get("status") != "OK" or not data.get("results"):
        raise MapsError(f"地名が見つかりませんでした（{data.get('status')}）。")

    top = data["results"][0]
    location = top["geometry"]["location"]
    return location["lat"], location["lng"], top.get("formatted_address", address)


def reverse_geocode(latitude, longitude):
    """緯度経度から地名を調べる。失敗したときは None を返す。"""
    if not has_api_key():
        return None

    params = {
        "latlng": f"{latitude},{longitude}",
        "language": "ja",
        "result_type": "locality|administrative_area_level_1",
        "key": API_KEY,
    }
    try:
        response = requests.get(GEOCODE_URL, params=params, timeout=TIMEOUT_SECONDS)
        data = response.json()
    except requests.RequestException:
        return None

    if data.get("status") != "OK" or not data.get("results"):
        return None

    return data["results"][0].get("formatted_address")


# ---------------------------------------------------------------
# スポット検索
# ---------------------------------------------------------------

def search_places(keyword, latitude, longitude, radius_m=3000, limit=5):
    """緯度経度のまわりで、キーワードに合うスポットを検索する。

    戻り値: スポット情報（辞書）のリスト
    """
    if not has_api_key():
        raise MapsError("APIキーが設定されていません。")
    if _denied_reason:
        raise MapsError(_denied_reason)

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": PLACES_FIELDS,
    }
    body = {
        "textQuery": keyword,
        "languageCode": "ja",
        "regionCode": "JP",
        "maxResultCount": limit,
        "locationBias": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                # APIの上限は50kmなので、それを超えないようにする
                "radius": float(min(radius_m, 50000)),
            }
        },
    }

    try:
        response = requests.post(
            PLACES_SEARCH_URL, headers=headers, json=body, timeout=TIMEOUT_SECONDS
        )
        data = response.json()
    except requests.RequestException as error:
        raise MapsError(f"通信に失敗しました: {error}")

    if "error" in data:
        error = data["error"]
        message = error.get("message", "不明なエラー")
        if error.get("code") in (401, 403, 429) or response.status_code in (401, 403, 429):
            raise remember_denied(f"{error.get('status', '')} {message}".strip())
        raise MapsError(f"スポット検索に失敗しました: {message}")

    return [to_spot(place, keyword) for place in data.get("places", [])]


def to_spot(place, keyword):
    """Places API の応答（1件）を、アプリで使いやすい辞書に変換する。"""
    return {
        "name": place.get("displayName", {}).get("text", "名前不明"),
        "address": place.get("formattedAddress", ""),
        "rating": place.get("rating"),
        "rating_count": place.get("userRatingCount", 0),
        "url": place.get("googleMapsUri", maps_search_url(keyword)),
        "keyword": keyword,
    }


# ---------------------------------------------------------------
# オフラインモード用（APIを使わずにGoogleマップの検索リンクを作る）
# ---------------------------------------------------------------

def maps_search_url(query, latitude=None, longitude=None):
    """Googleマップの検索リンクを作る（APIキー不要）。"""
    quoted = urllib.parse.quote(query)
    if latitude is not None and longitude is not None:
        return f"https://www.google.com/maps/search/{quoted}/@{latitude},{longitude},14z"
    return f"https://www.google.com/maps/search/?api=1&query={quoted}"


def fallback_spot(keyword, area_name, latitude=None, longitude=None):
    """APIが使えないときの代わりのスポット情報（Googleマップの検索リンク）を作る。

    緯度経度が分かっているときは、その地点を中心にキーワードだけで検索する。
    分からないときだけ「地名 キーワード」で検索する。
    """
    if latitude is not None and longitude is not None:
        return {
            "name": f"このあたりの{keyword}",
            "address": "Googleマップで探す",
            "rating": None,
            "rating_count": 0,
            "url": maps_search_url(keyword, latitude, longitude),
            "keyword": keyword,
            "source": "link",
        }

    return {
        "name": f"{area_name}の{keyword}",
        "address": "Googleマップで探す",
        "rating": None,
        "rating_count": 0,
        "url": maps_search_url(f"{area_name} {keyword}"),
        "keyword": keyword,
        "source": "link",
    }


def check_status():
    """APIキーが実際に使えるかどうかを調べる。

    戻り値: (使えるか, 説明の文字列)
    """
    if not has_api_key():
        return False, "APIキーが未設定です"

    try:
        geocode("東京駅")
    except MapsError as error:
        return False, str(error)

    return True, "Google Maps 連携中"
