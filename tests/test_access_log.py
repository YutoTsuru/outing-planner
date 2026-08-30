"""構造化アクセスログ（access_log.py）のテスト。

Flask のテストクライアントでリクエストを送り、logging に流れた行が
1行1JSONで、必要な項目を持っていることを確かめる。標準出力そのものは
見ず、ロガーにハンドラを差し替えて捕まえる。
"""

import json
import logging
import os

import pytest
from flask import Flask

import access_log
from outing_ml.config import CONFIG

MODELS_READY = os.path.exists(CONFIG.paths.category_model)
needs_models = pytest.mark.skipif(
    not MODELS_READY, reason="python train_all.py を先に実行してください"
)


@pytest.fixture
def captured_logs(monkeypatch):
    """access_log のロガーが出した行を集めるリストを返す。"""
    records: list[str] = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("outing_planner.access")
    # install() は「ハンドラが1つも無いときだけ」設定する実装なので、
    # ここでハンドラとレベルの両方をテスト用に差し替える
    # （レベルを直さないと、既定の WARNING で info() が捨てられてしまう）
    logger.handlers.clear()
    logger.addHandler(ListHandler())
    logger.setLevel(logging.INFO)
    yield records
    logger.handlers.clear()


def make_app() -> Flask:
    app = Flask(__name__)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    access_log.install(app)
    return app


def test_リクエストごとに1行のJSONが出る(captured_logs):
    make_app().test_client().get("/ping")

    assert len(captured_logs) == 1
    entry = json.loads(captured_logs[0])   # JSONとして読めなければここで例外
    assert entry["method"] == "GET"
    assert entry["path"] == "/ping"
    assert entry["status"] == 200
    assert entry["duration_ms"] >= 0
    assert entry["request_id"]
    assert entry["remote_addr"]


def test_クエリ文字列も記録される(captured_logs):
    make_app().test_client().get("/ping?a=1&b=2")

    entry = json.loads(captured_logs[0])
    assert entry["path"] == "/ping?a=1&b=2"


def test_レスポンスにリクエストIDのヘッダーが付く(captured_logs):
    response = make_app().test_client().get("/ping")

    entry = json.loads(captured_logs[0])
    assert response.headers["X-Request-Id"] == entry["request_id"]


def test_環境変数で止められる(captured_logs, monkeypatch):
    monkeypatch.setenv(access_log.ENV_FLAG, "0")

    make_app().test_client().get("/ping")

    assert captured_logs == []


def test_レート制限のヘッダーがあれば記録に含める(captured_logs):
    app = Flask(__name__)

    @app.get("/limited")
    def limited():
        response = app.make_response({"ok": True})
        response.headers["X-RateLimit-Remaining"] = "42"
        return response

    access_log.install(app)
    app.test_client().get("/limited")

    entry = json.loads(captured_logs[0])
    assert entry["rate_limit_remaining"] == 42


def test_レート制限のヘッダーが無ければ項目自体が無い(captured_logs):
    make_app().test_client().get("/ping")

    entry = json.loads(captured_logs[0])
    assert "rate_limit_remaining" not in entry


@needs_models
def test_実際のAPIアプリでも記録される(captured_logs):
    from api import create_api

    create_api().test_client().get("/api/health")

    assert len(captured_logs) == 1
    entry = json.loads(captured_logs[0])
    assert entry["path"] == "/api/health"


@needs_models
def test_実際のWebアプリでも記録される(captured_logs):
    from webapp import create_web_app

    create_web_app().test_client().get("/")

    assert len(captured_logs) == 1
