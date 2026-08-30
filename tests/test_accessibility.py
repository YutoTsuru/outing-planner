"""アクセシビリティに関わるマークアップのテスト。

見た目や動作ではなく「必要な属性が実際に出力されているか」を機械的に確かめる。
完璧なアクセシビリティ監査の代わりにはならないが、
「label が無い input」「role の無いエラー」のような明らかな欠落を検出できる。
"""

import os

import pytest

from outing_ml.config import CONFIG

MODELS_READY = os.path.exists(CONFIG.paths.category_model)
needs_models = pytest.mark.skipif(
    not MODELS_READY, reason="python train_all.py を先に実行してください"
)


@pytest.fixture
def client():
    from outing_ml.serve import OutingService
    from webapp import create_web_app

    app = create_web_app(outing=OutingService.load(), forecast=None)
    app.config.update(TESTING=True)
    return app.test_client()


def text_of(response) -> str:
    return response.get_data(as_text=True)


@needs_models
def test_トップ画面にスキップリンクとメインランドマークがある(client):
    body = text_of(client.get("/"))

    assert 'class="skip-link"' in body
    assert 'href="#main-content"' in body
    assert 'id="main-content"' in body
    assert "<main" in body


@needs_models
def test_ナビゲーションにラベルと現在地の印がある(client):
    body = text_of(client.get("/"))

    assert 'aria-label="主要ナビゲーション"' in body
    assert 'aria-current="page"' in body


@needs_models
def test_入力欄はすべてlabelと関連付いている(client):
    body = text_of(client.get("/"))

    # id="xxx" と for="xxx" の組がそろっているかを、雑にだが機械的に確かめる
    import re

    ids = set(re.findall(r'<(?:input|select)[^>]*\bid="([^"]+)"', body))
    labels_for = set(re.findall(r'<label[^>]*\bfor="([^"]+)"', body))

    assert ids, "input/select が1つも見つからない"
    missing = ids - labels_for
    assert not missing, f"label が無い入力欄: {missing}"


@needs_models
def test_予測結果の絵文字はaria_hiddenで隠れている(client):
    body = text_of(client.post("/predict", data={
        "temperature": "22", "rain_probability": "10",
        "wind_speed": "2", "humidity": "50",
    }))

    assert 'class="result-emoji" aria-hidden="true"' in body


@needs_models
def test_範囲外の入力の警告にrole_statusがある(client):
    body = text_of(client.post("/predict", data={
        "temperature": "99", "rain_probability": "10",
        "wind_speed": "2", "humidity": "50",
    }))

    assert 'class="warn" role="status"' in body


@needs_models
def test_エラー画面にrole_alertがある(client):
    response = client.get("/nothing-here")

    assert response.status_code == 404
    assert 'role="alert"' in text_of(response)


@needs_models
def test_モデルの状態画面の表にcaptionとscopeがある(client):
    body = text_of(client.get("/models"))

    assert 'class="sr-only"' in body
    assert body.count('scope="col"') >= 5


@needs_models
def test_予測の記録画面の表にcaptionとscopeがある(client):
    body = text_of(client.get("/history"))

    assert 'class="sr-only"' in body
    assert 'scope="col"' in body
