"""
お出かけプランナー：REST API（Flask）

画面（Gradio）を通さずに、学習したモデルを HTTP から使えるようにします。
他のアプリから呼んだり、まとめて予測したり、動作確認を自動化したりするためのものです。

    GET  /api/health          読み込めているモデルと版
    GET  /api/models          学習の履歴（版・成績・データの指紋）
    GET  /api/cities          翌日予報を出せる都市
    GET  /api/weather-types   天気タイプの一覧
    GET  /api/history         これまでの予測の記録と傾向
    POST /api/predict         天気4項目 → おすすめ・日和度・天気タイプ
    POST /api/predict/batch   まとめて予測（最大100件）
    GET  /api/forecast        あしたの天気と、そのおすすめ
    POST /api/plan            時間つきのお出かけプラン

実行方法:
    python api.py                 # http://127.0.0.1:5000 で起動
    python app.py                 # 画面と一緒に起動（/api で同じAPIが使える）
"""

import os
from typing import Any

from flask import Blueprint, Flask, jsonify, request

import geocoding
import planner
import prediction_log
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

def build_blueprint(outing: OutingService, forecast: ForecastService | None) -> Blueprint:
    """API のルートをまとめた Blueprint を作る。"""
    api = Blueprint("api", __name__)

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
        return jsonify({"ok": True, "category": category, "area": area_name,
                        "plan_markdown": text})

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
