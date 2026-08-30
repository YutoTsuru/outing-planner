"""
お出かけプランナー：REST API（Flask）

画面を通さずに、学習したモデルを HTTP から使えるようにします。
他のアプリから呼んだり、まとめて予測したり、動作確認を自動化したりするためのものです。

    GET  /api/health          読み込めているモデルと版
    GET  /api/models          学習の履歴（版・成績・データの指紋）
    GET  /api/cities          翌日予報を出せる都市
    GET  /api/weather-types   天気タイプの一覧
    GET  /api/history         これまでの予測の記録と傾向
    GET  /api/openapi.json    このAPIの仕様（OpenAPI 3.0）
    GET  /api/monitor         学習データと最近の入力のずれ（ドリフト監視）
    GET  /api/compare         全都市のあしたの予報を日和度が高い順に比較
    POST /api/predict         天気4項目 → おすすめ・日和度・天気タイプ
    POST /api/predict/batch   まとめて予測（最大100件）
    GET  /api/forecast        あしたの天気と、そのおすすめ
    GET  /api/week            数日先までの天気とおすすめ（再帰予測）
    POST /api/plan            時間つきのお出かけプラン（共有リンクも返す）
    GET  /api/share/<id>      共有リンクからプランを取り出す

実行方法:
    python api.py                 # http://127.0.0.1:5000 で起動
    python webapp.py              # 画面と一緒に起動（/api で同じAPIが使える）
"""

import os
from typing import Any

from flask import Blueprint, Flask, g, jsonify, request

import access_log
import city_comparison
import geocoding
import planner
import prediction_log
import rate_limit
import shared_plans
from monitoring import MonitorUnavailableError, monitor_report
from openapi_spec import build_spec as build_openapi_spec
from outing_ml import forecasting
from outing_ml.config import CATEGORIES, FEATURE_COLUMNS, INPUT_RANGES
from outing_ml.forecasting import ForecastService, ForecastUnavailableError
from outing_ml.registry import Registry
from outing_ml.serve import InvalidInputError, OutingService
from outing_ml.weather_source import UnknownCityError, city_names

# まとめて予測できる最大件数（大きなリクエストでサーバを詰まらせないため）
MAX_BATCH_SIZE = 100


class ApiError(Exception):
    """呼び出し側の間違いを、そのまま JSON で返すための例外。"""

    def __init__(self, message: str, status: int = 400, details: Any = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.details = details


# ---------------------------------------------------------------
# 入力の受け取り
# ---------------------------------------------------------------

def json_body() -> dict:
    """リクエストの JSON を取り出す。JSON でなければ 400。"""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ApiError("本文を JSON オブジェクトで送ってください", 400)
    return body


def weather_from(body: dict) -> dict:
    """天気4項目を取り出す。足りない項目があれば、何が足りないかを返す。"""
    missing = [column for column in FEATURE_COLUMNS if column not in body]
    if missing:
        raise ApiError(
            "天気の項目が足りません",
            400,
            {"missing": missing, "required": FEATURE_COLUMNS, "ranges": readable_ranges()},
        )
    return {column: body[column] for column in FEATURE_COLUMNS}


def readable_ranges() -> dict:
    """入力できる範囲（エラー時の案内に使う）。"""
    return {name: {"min": low, "max": high} for name, (low, high) in INPUT_RANGES.items()}


def query_int(name: str, default: int, minimum: int, maximum: int) -> int:
    """クエリ文字列から整数を読む（範囲外はエラー）。"""
    raw = request.args.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ApiError(f"{name} は整数で指定してください（受け取った値: {raw}）", 400) from None
    if not minimum <= value <= maximum:
        raise ApiError(f"{name} は {minimum}〜{maximum} で指定してください", 400)
    return value


def log_prediction(kind: str, payload: dict) -> None:
    """予測の記録（実装は prediction_log にある）。"""
    prediction_log.append(kind, payload)


# ---------------------------------------------------------------
# ルート
# ---------------------------------------------------------------

# エンドポイントごとの1分あたりの上限。載っていないものは DEFAULT_LIMIT_PER_MINUTE。
#
# forecast/week/compare/plan は外部（Open-Meteo・OpenStreetMap・Google Maps）へ
# 通信するため、predict などの純粋な計算だけのエンドポイントより低くしている。
_ENDPOINT_LIMITS = {
    "api.get_forecast": 20,
    "api.get_week": 15,
    "api.compare_cities": 30,
    "api.plan": 20,
}

# レート制限をかけないエンドポイント（起動確認・仕様の取得は絞る意味が薄い）
_RATE_LIMIT_EXEMPT = {"api.health", "api.openapi_json"}

# /api/compare?refresh=1 は、30分キャッシュを無視して47都市ぶんを
# 取得し直す重い操作。連発されると外部の利用上限に当たりかねないので、
# 上のエンドポイント単位の制限とは別に、もっと厳しい上限を重ねてかける。
_COMPARE_REFRESH_LIMIT_PER_MINUTE = 2


def _rate_limit_key(suffix: str = "") -> str:
    """レート制限のキー（IPアドレス＋エンドポイント＋任意の区別）を作る。

    プロキシ越しでは remote_addr が全員同じになりうるが、
    このアプリはローカル・教材用途のため、そこまでは対応していない。
    """
    identifier = request.remote_addr or "unknown"
    return f"{identifier}:{request.endpoint}:{suffix}"


def _too_many_requests(result) -> tuple:
    """429 のレスポンスを組み立てる。"""
    payload = {
        "ok": False,
        "error": {
            "message": "リクエストが多すぎます。しばらく待ってから試してください。",
            "details": {"limit_per_minute": result.limit,
                        "retry_after_seconds": result.reset_seconds},
        },
    }
    response = jsonify(payload)
    response.status_code = 429
    response.headers["Retry-After"] = str(result.reset_seconds)
    return response


def build_blueprint(outing: OutingService, forecast: ForecastService | None) -> Blueprint:
    """API のルートをまとめた Blueprint を作る。"""
    api = Blueprint("api", __name__)

    @api.before_request
    def enforce_rate_limit():
        """エンドポイントごとの上限を確認する。超過なら429で止める。"""
        if request.endpoint in _RATE_LIMIT_EXEMPT:
            return None

        limit = _ENDPOINT_LIMITS.get(request.endpoint, rate_limit.DEFAULT_LIMIT_PER_MINUTE)
        result = rate_limit.check(_rate_limit_key(), limit=limit)
        if not result.allowed:
            return _too_many_requests(result)

        g.rate_limit_result = result

        if request.endpoint == "api.compare_cities" and request.args.get("refresh") == "1":
            refresh_result = rate_limit.check(
                _rate_limit_key("refresh"), limit=_COMPARE_REFRESH_LIMIT_PER_MINUTE
            )
            if not refresh_result.allowed:
                return _too_many_requests(refresh_result)

        return None

    @api.after_request
    def add_rate_limit_headers(response):
        """あと何回使えるかを、成功したレスポンスにも添える。"""
        result = g.get("rate_limit_result")
        if result is not None:
            response.headers["X-RateLimit-Limit"] = str(result.limit)
            response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        return response

    @api.get("/openapi.json")
    def openapi_json():
        """このAPIの仕様（OpenAPI 3.0）を返す。"""
        return jsonify(build_openapi_spec())

    @api.get("/health")
    def health():
        """読み込めているモデルと版を返す（起動確認用）。"""
        return jsonify({"ok": True, **outing.health(),
                        "forecast": forecast.bundle.version if forecast else None})

    @api.get("/models")
    def models():
        """学習の履歴を返す（どのデータで・いつ・どの版を作ったか）。"""
        registry = Registry()
        entries = []
        for name in ("category-classifier", "comfort-regressor",
                     "next-day-forecast", "weather-type-clustering"):
            latest = registry.latest(name)
            if latest:
                entries.append({
                    "model_name": latest["model_name"],
                    "version": latest["version"],
                    "task": latest["task"],
                    "created_at": latest["created_at"],
                    "git_sha": latest["git_sha"],
                    "data_sha256": latest["data"].get("sha256"),
                    "metrics": latest["metrics"],
                })
        return jsonify({"ok": True, "models": entries})

    @api.get("/cities")
    def cities():
        """翌日予報を出せる都市の一覧。"""
        return jsonify({"ok": True, "cities": city_names()})

    @api.get("/weather-types")
    def weather_types():
        """天気タイプの一覧（中心の値・件数・名前）。"""
        if outing.weather_type is None:
            raise ApiError("天気タイプ分けモデルが読み込まれていません", 503)

        names = outing.weather_type.metadata.get("cluster_names", {})
        return jsonify({
            "ok": True,
            "version": outing.weather_type.version,
            "types": [{"id": int(key), "name": value} for key, value in sorted(names.items())],
        })

    @api.post("/predict")
    def predict():
        """天気4項目から、おすすめ・日和度・天気タイプを返す。"""
        body = json_body()
        weather = weather_from(body)

        try:
            result = outing.predict(**weather)
        except InvalidInputError as error:
            raise ApiError(str(error), 400, {"ranges": readable_ranges()}) from error

        log_prediction("predict", {"input": weather, "category": result.category,
                                   "confidence": result.confidence})
        return jsonify({"ok": True, "input": weather, **result.to_dict()})

    @api.post("/predict/batch")
    def predict_batch():
        """まとめて予測する。"""
        body = json_body()
        days = body.get("days")

        if not isinstance(days, list) or not days:
            raise ApiError("days に天気の配列を入れてください", 400)
        if len(days) > MAX_BATCH_SIZE:
            raise ApiError(f"一度に predict できるのは {MAX_BATCH_SIZE} 件までです", 400,
                           {"received": len(days)})

        results = []
        for index, day in enumerate(days):
            if not isinstance(day, dict):
                raise ApiError(f"{index} 番目が JSON オブジェクトではありません", 400)

            # 何件目が悪いのかが分からないと、送り直す側が原因を探せない
            try:
                weather = weather_from(day)
            except ApiError as error:
                raise ApiError(f"{index} 番目: {error.message}", 400, error.details) from error

            try:
                result = outing.predict(**weather)
            except InvalidInputError as error:
                raise ApiError(f"{index} 番目: {error}", 400,
                               {"ranges": readable_ranges()}) from error
            results.append({"input": weather, **result.to_dict()})

        log_prediction("predict_batch", {"count": len(results)})
        return jsonify({"ok": True, "count": len(results), "results": results})

    @api.get("/week")
    def get_week():
        """指定した都市の、数日先までの天気とおすすめをまとめて返す。"""
        if forecast is None:
            raise ApiError("翌日予測モデルが読み込まれていません", 503)

        city = request.args.get("city")
        if not city:
            raise ApiError("city を指定してください", 400, {"cities": city_names()})

        days_ahead = query_int("days", 7, 1, forecasting.MAX_FORECAST_DAYS)
        history_days = query_int("history_days", 10, 5, 30)

        try:
            result = forecast.predict_week(city, days_ahead=days_ahead,
                                           history_days=history_days)
        except UnknownCityError as error:
            raise ApiError(str(error), 400, {"cities": city_names()}) from error
        except ForecastUnavailableError as error:
            raise ApiError(str(error), 503) from error

        log_prediction("week", {"city": city, "days": len(result.days)})
        return jsonify({"ok": True, **result.to_dict()})

    @api.get("/forecast")
    def get_forecast():
        """あしたの天気を予測して、そのおすすめまで返す。"""
        if forecast is None:
            raise ApiError("翌日予測モデルが読み込まれていません", 503)

        city = request.args.get("city")
        if not city:
            raise ApiError("city を指定してください", 400, {"cities": city_names()})

        days = query_int("days", 12, 5, 30)

        try:
            result = forecast.predict_tomorrow(city, days=days)
        except UnknownCityError as error:
            raise ApiError(str(error), 400, {"cities": city_names()}) from error
        except ForecastUnavailableError as error:
            raise ApiError(str(error), 503) from error

        log_prediction("forecast", {
            "city": city,
            "target_date": result.target_date,
            "weather": result.weather,
            "category": result.recommendation.category if result.recommendation else None,
            "confidence": result.recommendation.confidence if result.recommendation else None,
        })
        return jsonify({"ok": True, **result.to_dict()})

    @api.get("/monitor")
    def get_monitor():
        """学習データと最近の入力を比べて、そろそろ学習し直したほうがよいかを見る。"""
        try:
            report = monitor_report()
        except MonitorUnavailableError as error:
            raise ApiError(str(error), 503) from error

        return jsonify({"ok": True, **report})

    @api.get("/compare")
    def compare_cities():
        """全都市の、あしたの予報を日和度が高い順にまとめて返す。

        初回（キャッシュが無いとき）は都市の数だけ外部へ通信するため、
        数十秒かかることがある。次回以降はキャッシュが効くので一瞬で返る。
        """
        if forecast is None:
            raise ApiError("翌日予測モデルが読み込まれていません", 503)

        force = request.args.get("refresh") == "1"
        data = city_comparison.get_comparison(forecast, force_refresh=force)
        return jsonify({"ok": True, **data})

    @api.get("/history")
    def history():
        """これまでの予測の記録と、その傾向を返す。

        どんな天気のときに使われたかが見えると、学習データの範囲と
        実際の使われ方がずれていないかを確かめられる。
        """
        limit = query_int("limit", prediction_log.DEFAULT_LIMIT, 1, 1000)
        entries = prediction_log.read_entries(limit=limit)

        return jsonify({
            "ok": True,
            "logging_enabled": prediction_log.enabled(),
            "summary": prediction_log.summarize(entries),
            "entries": entries,
        })

    @api.post("/plan")
    def plan():
        """時間つきのお出かけプランを作る（Markdown で返す）。"""
        body = json_body()

        category = body.get("category")
        if category not in CATEGORIES:
            raise ApiError("category が正しくありません", 400, {"allowed": CATEGORIES})

        start_time = body.get("start_time", "10:00")
        end_time = body.get("end_time", "17:00")
        radius_km = body.get("radius_km", 3)
        if not isinstance(radius_km, (int, float)) or not 1 <= radius_km <= 20:
            raise ApiError("radius_km は 1〜20 で指定してください", 400)

        latitude, longitude = body.get("latitude"), body.get("longitude")
        area_name = body.get("area")

        if latitude is None or longitude is None:
            if not area_name:
                raise ApiError("area か、latitude と longitude を指定してください", 400)
            # 地名から緯度経度を引く（Google → OpenStreetMap の順に試す）
            try:
                latitude, longitude, area_name = geocoding.resolve_area(area_name)
            except geocoding.AreaNotFoundError as error:
                raise ApiError(str(error), 404) from error

        wishes = body.get("wishes") or []
        if not isinstance(wishes, list):
            raise ApiError("wishes は文字列の配列で指定してください", 400)

        text = planner.build_plan(
            category, float(latitude), float(longitude), area_name or "指定の場所",
            start_time, end_time, radius_m=int(radius_km * 1000),
            wishes=[str(wish) for wish in wishes[:planner.MAX_WISHES]],
        )
        plan_id = shared_plans.save(category, area_name or "指定の場所", text)
        return jsonify({"ok": True, "category": category, "area": area_name,
                        "plan_markdown": text, "share_id": plan_id,
                        "share_url": f"/share/{plan_id}"})

    @api.get("/share/<plan_id>")
    def get_shared_plan(plan_id):
        """共有リンクからプランを取り出す。"""
        try:
            entry = shared_plans.find(plan_id)
        except shared_plans.PlanNotFoundError:
            raise ApiError("そのプランは見つかりませんでした", 404) from None

        return jsonify({"ok": True, **entry})

    return api


# ---------------------------------------------------------------
# アプリの組み立て
# ---------------------------------------------------------------

def create_api(outing: OutingService = None, forecast: ForecastService = None,
               load_forecast: bool = True) -> Flask:
    """Flask アプリを作る。

    テストからはモデルを差し替えられるように、サービスを引数で受け取れるようにしています。
    """
    app = Flask(__name__)
    app.json.ensure_ascii = False   # 日本語をそのまま返す
    access_log.install(app)

    outing = outing or OutingService.load()

    if forecast is None and load_forecast:
        try:
            forecast = ForecastService.load(outing=outing)
        except FileNotFoundError:
            forecast = None   # 翌日予測モデルが無くても、ほかの機能は使える

    app.register_blueprint(build_blueprint(outing, forecast), url_prefix="/api")

    @app.errorhandler(ApiError)
    def handle_api_error(error: ApiError):
        payload = {"ok": False, "error": {"message": error.message}}
        if error.details is not None:
            payload["error"]["details"] = error.details
        return jsonify(payload), error.status

    @app.errorhandler(404)
    def handle_not_found(_error):
        return jsonify({"ok": False, "error": {"message": "そのURLはありません"}}), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(_error):
        return jsonify({"ok": False, "error": {"message": "そのHTTPメソッドは使えません"}}), 405

    @app.errorhandler(Exception)
    def handle_unexpected(error: Exception):
        # 中身をそのまま返すと内部構造が漏れるので、種類だけ伝える
        app.logger.exception("想定外のエラー", exc_info=error)
        return jsonify({"ok": False, "error": {"message": "サーバ側で問題が起きました"}}), 500

    return app


def main():
    port = int(os.environ.get("PORT", "5000"))
    print(f"REST API を起動します: http://127.0.0.1:{port}/api/health")
    create_api().run(host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
