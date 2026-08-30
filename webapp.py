"""
お出かけプランナー：Web アプリ（Flask）

画面（HTML）と REST API を1つのサーバでまとめて提供します。

    /              天気を入力する画面
    /predict       予測結果（POST）
    /forecast      都市を選んで、あしたの予報から予測する
    /plan          時間つきのお出かけプラン（POST）
    /models        いま動いているモデルの版と成績
    /api/...       同じ機能を JSON で返す REST API（api.py）

Gradio 版（app.py）と同じモデル・同じ文言を使っています。
違うのは見た目だけで、予測はどちらも outing_ml.serve を通ります。

実行方法:
    python webapp.py                    # http://127.0.0.1:5000
    PORT=8000 python webapp.py          # ポートを変える
"""

import json
import os

from flask import Blueprint, Flask, redirect, render_template, request, url_for

import api as api_module
import city_comparison
import geocoding
import monitoring
import prediction_log
import presentation
from outing_ml.config import CATEGORIES, CONFIG, FEATURE_COLUMNS
from outing_ml.forecasting import ForecastService, ForecastUnavailableError
from outing_ml.registry import Registry
from outing_ml.serve import InvalidInputError, OutingService
from outing_ml.weather_source import UnknownCityError, city_names

# 画面の表で見せる、モデルごとの代表的な成績
HEADLINE_METRICS = {
    "category-classifier": [("正解率", "test_accuracy", "{:.3f}"),
                            ("未来データ", "holdout_accuracy", "{:.3f}")],
    "comfort-regressor": [("MAE", "test_mae", "{:.2f}点"),
                          ("下限", "noise_floor_mae", "{:.2f}点")],
    "next-day-forecast": [("気温MAE", "walk_forward_temperature_mae", "{:.2f}℃")],
    "weather-type-clustering": [("タイプ数", "n_clusters", "{:.0f}"),
                                ("シルエット", "silhouette", "{:.3f}")],
}


class WebError(Exception):
    """画面向けのエラー（そのままエラーページにする）。"""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


def read_weather(form) -> dict:
    """フォームから天気4項目を読む。"""
    weather = {}
    for column in FEATURE_COLUMNS:
        raw = form.get(column)
        if raw is None or raw == "":
            raise WebError(f"{column} が入力されていません")
        try:
            weather[column] = float(raw)
        except ValueError:
            raise WebError(f"{column} は数値で入力してください（入力: {raw}）") from None
    return weather


def sorted_probabilities(result) -> list:
    """確率を高い順に並べる（棒グラフ用）。"""
    return sorted(result.probabilities.items(), key=lambda item: item[1], reverse=True)


def result_context(result, weather, area=None) -> dict:
    """予測結果の画面に渡すもの。"""
    return {
        "result": result,
        "label": presentation.label_view(result.category),
        "labels": presentation.label_views(),
        "probabilities": sorted_probabilities(result),
        "reason": presentation.build_reason(result.category, **weather),
        "fields": presentation.weather_fields(),
        "area": area,
    }


def registry_rows() -> list:
    """学習の履歴を、画面の表に出せる形にする。"""
    registry = Registry()
    rows = []

    for name in ("category-classifier", "comfort-regressor",
                 "next-day-forecast", "weather-type-clustering"):
        entry = registry.latest(name)
        if not entry:
            continue

        headline = []
        for label, key, form in HEADLINE_METRICS.get(name, []):
            value = entry["metrics"].get(key)
            if isinstance(value, (int, float)):
                headline.append((label, form.format(value)))

        rows.append({**entry, "headline": headline})

    return rows


def weather_type_rows() -> list:
    """天気タイプの一覧（中心の値と割合）。"""
    path = CONFIG.paths.weather_type_card
    if not os.path.exists(path):
        return []

    with open(path, encoding="utf-8") as file:
        card = json.load(file)
    return sorted(card.get("clusters", []), key=lambda item: item["id"])


def build_web_blueprint(outing: OutingService, forecast: ForecastService | None) -> Blueprint:
    """画面のルートをまとめた Blueprint を作る。"""
    web = Blueprint("web", __name__)

    @web.get("/")
    def index():
        return render_template("index.html",
                               fields=presentation.weather_fields(),
                               cities=city_names())

    @web.post("/predict")
    def predict_page():
        weather = read_weather(request.form)
        try:
            result = outing.predict(**weather)
        except InvalidInputError as error:
            raise WebError(str(error)) from error

        prediction_log.append("web_predict", {"input": weather,
                                              "category": result.category,
                                              "confidence": result.confidence})
        return render_template("predict.html", weather=weather,
                               **result_context(result, weather))

    @web.get("/forecast")
    def forecast_page():
        city = request.args.get("city")
        if not city:
            return redirect(url_for("web.index"))
        if forecast is None:
            raise WebError("翌日予測モデルが読み込まれていません。python train_forecast.py を実行してください。", 503)

        try:
            result = forecast.predict_tomorrow(city)
        except UnknownCityError as error:
            raise WebError(str(error)) from error
        except ForecastUnavailableError as error:
            raise WebError(str(error), 503) from error

        prediction_log.append("web_forecast", {
            "city": city,
            "target_date": result.target_date,
            "weather": result.weather,
            "category": result.recommendation.category if result.recommendation else None,
            "confidence": result.recommendation.confidence if result.recommendation else None,
        })
        return render_template("forecast.html", city=city, forecast=result,
                               **result_context(result.recommendation, result.weather))

    @web.post("/plan")
    def plan_page():
        import planner

        category = request.form.get("category")
        if category not in CATEGORIES:
            raise WebError("カテゴリが正しくありません")

        area = (request.form.get("area") or "").strip()
        if not area:
            raise WebError("場所を入力してください（例：京都市）")

        try:
            radius_km = int(request.form.get("radius_km", 3))
        except ValueError:
            raise WebError("探す範囲は数値で入力してください") from None
        if not 1 <= radius_km <= 20:
            raise WebError("探す範囲は 1〜20 km で指定してください")

        try:
            latitude, longitude, area_name = geocoding.resolve_area(area)
        except geocoding.AreaNotFoundError as error:
            raise WebError(f"{error}。市区町村名で入れてみてください。", 404) from error

        text = planner.build_plan(
            category, latitude, longitude, area_name,
            request.form.get("start_time", "10:00"),
            request.form.get("end_time", "17:00"),
            radius_m=radius_km * 1000,
        )
        return render_template("plan.html", area=area_name, category=category,
                               plan_text=text, labels=presentation.label_views())

    @web.get("/monitor")
    def monitor_page():
        try:
            report = monitoring.monitor_report()
        except monitoring.MonitorUnavailableError as error:
            raise WebError(str(error), 503) from error

        return render_template("monitor.html", report=report,
                               labels=presentation.label_views(),
                               fields_view=presentation.weather_fields())

    @web.get("/compare")
    def compare_page():
        if forecast is None:
            raise WebError("翌日予測モデルが読み込まれていません。python train_forecast.py を実行してください。", 503)

        force = request.args.get("refresh") == "1"
        data = city_comparison.get_comparison(forecast, force_refresh=force)
        return render_template("compare.html", data=data,
                               labels=presentation.label_views(),
                               fields_view=presentation.weather_fields())

    @web.get("/history")
    def history_page():
        limit = 50
        raw = request.args.get("limit")
        if raw:
            try:
                limit = max(1, min(500, int(raw)))
            except ValueError:
                raise WebError("limit は数値で指定してください") from None

        entries = prediction_log.read_entries(limit=limit)
        return render_template(
            "history.html",
            entries=entries,
            summary=prediction_log.summarize(entries),
            labels=presentation.label_views(),
            fields_view=presentation.weather_fields(),
            logging_enabled=prediction_log.enabled(),
            log_path=prediction_log.LOG_PATH,
            limit=limit,
        )

    @web.get("/models")
    def models_page():
        return render_template("models.html", health=outing.health(),
                               models=registry_rows(),
                               weather_types=weather_type_rows())

    return web


def create_web_app(outing: OutingService = None,
                   forecast: ForecastService = None) -> Flask:
    """画面と REST API をまとめた Flask アプリを作る。"""
    app = Flask(__name__)
    app.json.ensure_ascii = False

    outing = outing or OutingService.load()
    if forecast is None:
        try:
            forecast = ForecastService.load(outing=outing)
        except FileNotFoundError:
            forecast = None

    app.register_blueprint(build_web_blueprint(outing, forecast))
    app.register_blueprint(api_module.build_blueprint(outing, forecast), url_prefix="/api")

    @app.errorhandler(WebError)
    def handle_web_error(error: WebError):
        return render_template("error.html", code=error.status,
                               message=error.message), error.status

    @app.errorhandler(api_module.ApiError)
    def handle_api_error(error: api_module.ApiError):
        # /api 以下は JSON、画面側は HTML で返す
        if request.path.startswith("/api"):
            from flask import jsonify

            payload = {"ok": False, "error": {"message": error.message}}
            if error.details is not None:
                payload["error"]["details"] = error.details
            return jsonify(payload), error.status
        return render_template("error.html", code=error.status,
                               message=error.message), error.status

    @app.errorhandler(404)
    def handle_not_found(_error):
        if request.path.startswith("/api"):
            from flask import jsonify

            return jsonify({"ok": False, "error": {"message": "そのURLはありません"}}), 404
        return render_template("error.html", code=404,
                               message="そのページはありません。"), 404

    @app.errorhandler(500)
    def handle_server_error(_error):
        return render_template("error.html", code=500,
                               message="サーバ側で問題が起きました。"), 500

    return app


def main():
    port = int(os.environ.get("PORT", "5000"))
    print(f"Web アプリを起動します: http://127.0.0.1:{port}/")
    print(f"REST API の確認:        http://127.0.0.1:{port}/api/health")
    create_web_app().run(host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
