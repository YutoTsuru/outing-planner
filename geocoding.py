"""地名から緯度経度を調べる。

Google Maps →（使えなければ）OpenStreetMap の順に試します。
画面（Gradio / Flask）と REST API の3か所から呼ばれるので、切り替えの順番と
「見つからなかったときの扱い」をここ1か所に置きます。

注意: maps_api.geocode も osm_api.geocode も、失敗すると None ではなく例外を投げます。
片方だけ try で囲むと、APIキーが無い環境で OpenStreetMap への切り替えが動きません。
"""

import maps_api
import osm_api


class AreaNotFoundError(ValueError):
    """地名から場所を決められなかったときに投げる例外。"""


def resolve_area(query: str) -> tuple[float, float, str]:
    """地名から (緯度, 経度, 正式な地名) を返す。

    Google Maps はAPIキーと請求先アカウントが要るので、無ければ黙って
    OpenStreetMap を使います（アプリはキー無しでも動く、が前提のため）。
    """
    query = (query or "").strip()
    if not query:
        raise AreaNotFoundError("市区町村・都道府県を入力してください（例：京都市）")

    # 1. Google Maps（キーがあり、まだ拒否されていないときだけ試す）
    if maps_api.has_api_key() and maps_api.denied_reason() is None:
        try:
            return maps_api.geocode(query)
        except maps_api.MapsError:
            pass

    # 2. OpenStreetMap（キー不要）
    try:
        return osm_api.geocode(query)
    except osm_api.OsmError as error:
        raise AreaNotFoundError(f"「{query}」が見つかりませんでした（{error}）") from error
