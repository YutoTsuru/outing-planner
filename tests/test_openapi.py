"""OpenAPI 仕様（openapi_spec.py）のテスト。

api.py の実際のルートと、手書きの仕様（openapi_spec.py）が食い違っていないかを
機械的に突き合わせる。仕様の中身（説明文など）までは検証しないが、
「そのエンドポイントが載っているか」「メソッドが合っているか」はここで守る。
"""

import os
import re

import pytest

from openapi_spec import build_spec
from outing_ml.config import CONFIG

MODELS_READY = os.path.exists(CONFIG.paths.category_model)
needs_models = pytest.mark.skipif(
    not MODELS_READY, reason="python train_all.py を先に実行してください"
)

# api.py から実際のルートを抜き出す（@api.get("/xxx") / @api.post("/xxx") の形）
ROUTE_PATTERN = re.compile(r'@api\.(get|post)\("([^"]+)"\)')


# 仕様自身を指す /openapi.json は、仕様の中に自己参照として載せる必要が無い
SELF_ROUTE = ("GET", "/openapi.json")


def actual_routes() -> set[tuple[str, str]]:
    """api.py に実際に定義されているルートを (メソッド, パス) の集合で返す（自己参照を除く）。"""
    with open("api.py", encoding="utf-8") as file:
        source = file.read()
    routes = {(method.upper(), path) for method, path in ROUTE_PATTERN.findall(source)}
    routes.discard(SELF_ROUTE)
    return routes


def spec_routes(spec: dict) -> set[tuple[str, str]]:
    """仕様に載っているルートを (メソッド, パス) の集合で返す。"""
    routes = set()
    for path, methods in spec["paths"].items():
        for method in methods:
            routes.add((method.upper(), path))
    return routes


def test_仕様はOpenAPIとして妥当():
    from openapi_spec_validator import validate

    validate(build_spec())   # 例外が出なければ合格


def test_基本情報が入っている():
    spec = build_spec()

    assert spec["openapi"].startswith("3.")
    assert spec["info"]["title"]
    assert spec["info"]["version"]


def test_全エンドポイントが仕様に載っている():
    spec = build_spec()

    missing = actual_routes() - spec_routes(spec)
    assert not missing, f"仕様に無いエンドポイント: {missing}"


def test_仕様に無いエンドポイントが書かれていない():
    """api.py に無いのに仕様にだけ残っている、削除し忘れを検出する。"""
    spec = build_spec()

    extra = spec_routes(spec) - actual_routes()
    assert not extra, f"api.py に無いのに仕様に残っている: {extra}"


def test_おもな入出力の型が書かれている():
    spec = build_spec()

    predict = spec["paths"]["/predict"]["post"]
    body_schema = predict["requestBody"]["content"]["application/json"]["schema"]
    assert set(body_schema["required"]) == {
        "temperature", "rain_probability", "wind_speed", "humidity",
    }


@needs_models
def test_APIから返る仕様が同じ内容():
    """/api/openapi.json が build_spec() と同じものを返すことを確かめる。"""
    from api import create_api

    client = create_api().test_client()
    body = client.get("/api/openapi.json").get_json()

    assert body == build_spec()
