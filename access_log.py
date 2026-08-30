"""構造化アクセスログ（1行1JSON、標準出力）。

Flask の開発サーバは既定でも「127.0.0.1 - - [日付] "GET /api/health" 200 -」
のような行を出しますが、これは人が読むための形で、機械的に集計しにくいものです。
ここでは同じ情報を JSON 1行にして出し、あとから jq などで絞り込めるようにします。

出す項目:
    request_id    リクエストごとの一意なID（UUID）。同じリクエストのログをたどれる
    timestamp     ISO 8601（UTC）
    method        HTTPメソッド
    path          パス（クエリ文字列を含む）
    status        レスポンスのステータスコード
    duration_ms   処理にかかった時間（ミリ秒）
    remote_addr   リクエスト元のIPアドレス
    rate_limit    レート制限の残り回数（かかっているエンドポイントのみ）

環境変数 OUTING_ACCESS_LOG=0 で出力を止められます（既定はオン）。
"""

import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime

from flask import Flask, g, request

# 出力を止めたいときの環境変数
ENV_FLAG = "OUTING_ACCESS_LOG"

_logger = logging.getLogger("outing_planner.access")


def enabled() -> bool:
    """アクセスログを出す設定になっているか。"""
    return os.environ.get(ENV_FLAG, "1") != "0"


def _configure_logger() -> None:
    """ロガーを1行1メッセージだけを出す形にする（Flask/Werkzeugの装飾を足さない）。"""
    if _logger.handlers:
        return   # 二重登録を防ぐ（同じプロセスで複数の Flask app を作ることがあるため）

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False   # ルートロガーへ二重に流さない


def install(app: Flask) -> None:
    """Flask アプリに、リクエストごとのアクセスログを仕込む。"""
    _configure_logger()

    @app.before_request
    def _start_timer():
        g.access_log_request_id = str(uuid.uuid4())
        g.access_log_started_at = time.perf_counter()

    @app.after_request
    def _write_log(response):
        if not enabled():
            return response

        started_at = g.get("access_log_started_at")
        duration_ms = (
            round((time.perf_counter() - started_at) * 1000, 2)
            if started_at is not None else None
        )

        entry = {
            "request_id": g.get("access_log_request_id"),
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "method": request.method,
            "path": request.full_path if request.query_string else request.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
            "remote_addr": request.remote_addr,
        }

        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining is not None:
            entry["rate_limit_remaining"] = int(remaining)

        response.headers["X-Request-Id"] = entry["request_id"]
        _logger.info(json.dumps(entry, ensure_ascii=False))
        return response
