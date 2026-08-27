"""地名の変換のテスト。

Google Maps も OpenStreetMap も、失敗すると None ではなく例外を投げます。
片方だけ try で囲むと、APIキーが無い環境で切り替えが動かず 500 になります
（実際にそれで /plan が落ちた）。切り替えが効くことを固定します。
"""

import pytest

import geocoding
import maps_api
import osm_api


@pytest.fixture(autouse=True)
def no_google(monkeypatch):
    """既定では「Google Maps のキーが無い」状態にする。"""
    monkeypatch.setattr(maps_api, "has_api_key", lambda: False)


def test_地名が空なら断る():
    with pytest.raises(geocoding.AreaNotFoundError):
        geocoding.resolve_area("   ")


def test_キーが無ければOpenStreetMapを使う(monkeypatch):
    monkeypatch.setattr(osm_api, "geocode", lambda q: (35.0, 135.7, "京都府京都市"))

    assert geocoding.resolve_area("京都市") == (35.0, 135.7, "京都府京都市")


def test_GoogleMapsが失敗したらOpenStreetMapに切り替える(monkeypatch):
    monkeypatch.setattr(maps_api, "has_api_key", lambda: True)
    monkeypatch.setattr(maps_api, "denied_reason", lambda: None)

    def google_fails(query):
        raise maps_api.MapsError("APIキーが設定されていません。")

    monkeypatch.setattr(maps_api, "geocode", google_fails)
    monkeypatch.setattr(osm_api, "geocode", lambda q: (34.7, 135.5, "大阪府大阪市"))

    assert geocoding.resolve_area("大阪市") == (34.7, 135.5, "大阪府大阪市")


def test_GoogleMapsが使えるならそちらを使う(monkeypatch):
    monkeypatch.setattr(maps_api, "has_api_key", lambda: True)
    monkeypatch.setattr(maps_api, "denied_reason", lambda: None)
    monkeypatch.setattr(maps_api, "geocode", lambda q: (35.6, 139.7, "東京都"))
    monkeypatch.setattr(osm_api, "geocode",
                        lambda q: pytest.fail("Google が使えるのに切り替わった"))

    assert geocoding.resolve_area("東京")[2] == "東京都"


def test_どちらでも見つからなければ分かるエラーにする(monkeypatch):
    def osm_fails(query):
        raise osm_api.OsmError("その地名は見つかりませんでした")

    monkeypatch.setattr(osm_api, "geocode", osm_fails)

    with pytest.raises(geocoding.AreaNotFoundError) as error:
        geocoding.resolve_area("ありえない地名")
    assert "ありえない地名" in str(error.value)
