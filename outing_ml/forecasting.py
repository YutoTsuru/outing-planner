"""翌日の天気予報と、そこから出すおすすめ。

学習した2つのモデルをつなげて使います。

    直近の実測（Open-Meteo） → 翌日予測モデル → あしたの天気
                                                → カテゴリ予測モデル → あしたのおすすめ

予報の誤差はそのままおすすめのズレになります（実測 2025〜2026 年で一致率 61.8%）。
そのため、予測した天気には必ず予測区間（だいたいこの範囲）を添えて返します。
"""

import time
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from outing_ml import features as feature_module
from outing_ml import weather_source
from outing_ml.config import CONFIG, FEATURE_COLUMNS, INPUT_RANGES
from outing_ml.registry import ModelBundle, load_bundle
from outing_ml.serve import OutingService, Recommendation


class ForecastUnavailableError(RuntimeError):
    """予報を出すのに必要な材料がそろわないときに投げる例外。"""


# 週間予報でさかのぼれる日数の上限。
# 1日ごとに予測を入力へ回す再帰予測のため、先に行くほど誤差が積み重なる。
# あまり先まで伸ばしても参考にならないので、上限を設けている。
MAX_FORECAST_DAYS = 7


@dataclass
class TomorrowForecast:
    """あしたの予報と、そこから出したおすすめ。"""

    city: str
    base_date: str                      # 予測のもとにした「いちばん新しい実測の日」
    target_date: str                    # 予測した日（＝そのつぎの日）
    weather: dict[str, float]           # 予測した天気4項目
    interval: dict[str, dict[str, float]] = field(default_factory=dict)
    recommendation: Recommendation | None = None
    model_version: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "city": self.city,
            "base_date": self.base_date,
            "target_date": self.target_date,
            "weather": self.weather,
            "interval": self.interval,
            "recommendation": (
                self.recommendation.to_dict() if self.recommendation else None
            ),
            "model_version": self.model_version,
            "notes": self.notes,
        }


@dataclass
class DailyForecast:
    """週間予報のうち、1日ぶんの予測。"""

    day: int                            # 1 = あした、2 = あさって、…
    date: str
    weather: dict[str, float]
    interval: dict[str, dict[str, float]] = field(default_factory=dict)
    recommendation: Recommendation | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "day": self.day,
            "date": self.date,
            "weather": self.weather,
            "interval": self.interval,
            "recommendation": (
                self.recommendation.to_dict() if self.recommendation else None
            ),
        }


@dataclass
class WeeklyForecast:
    """複数日ぶんの予報をまとめたもの。"""

    city: str
    base_date: str
    days: list[DailyForecast] = field(default_factory=list)
    model_version: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "city": self.city,
            "base_date": self.base_date,
            "days": [day.to_dict() for day in self.days],
            "model_version": self.model_version,
            "notes": self.notes,
        }


class ForecastService:
    """翌日予測モデルを使って、あしたの天気とおすすめを返す。"""

    def __init__(self, bundle: ModelBundle, outing: OutingService = None,
                 source=weather_source):
        self.bundle = bundle
        self.outing = outing
        self.source = source

    @classmethod
    def load(cls, outing: OutingService = None) -> "ForecastService":
        """成果物を読み込んでサービスを作る。"""
        return cls(bundle=load_bundle(CONFIG.paths.forecast_model), outing=outing)

    # -----------------------------------------------------------
    # 予測
    # -----------------------------------------------------------

    def predict_tomorrow(self, city: str, days: int = 10) -> TomorrowForecast:
        """指定した都市の、あしたの天気とおすすめを返す。"""
        recent = self.source.recent_daily(city, days=days)
        if recent.empty:
            raise ForecastUnavailableError(
                f"{city} の直近の実測データが取得できませんでした。時間をおいて試してください。"
            )

        frame = feature_module.build_prediction_frame(recent)
        if frame.empty:
            raise ForecastUnavailableError(
                f"{city} の直近データが足りません（3日ぶん以上そろっている必要があります）。"
            )

        row = frame.iloc[[-1]]
        base_date = pd.to_datetime(row["date"].iloc[0])
        inputs = row[self.bundle.feature_names]

        weather = self._point_forecast(inputs)
        interval = self._interval_forecast(inputs)

        forecast = TomorrowForecast(
            city=city,
            base_date=base_date.date().isoformat(),
            target_date=(base_date + timedelta(days=1)).date().isoformat(),
            weather=weather,
            interval=interval,
            model_version=self.bundle.version,
        )

        # 実測は数日おくれて公開されるので、「あした」が今日より前になることがある
        forecast.notes.append(
            "実測データは数日おくれて公開されるため、予測の起点は直近の入手可能日です。"
        )
        forecast.notes.append(
            "予測区間は想定80%ですが、実測では 68.9〜89.0% と項目によってずれます"
            "（doc/forecast.md 参照）。"
        )

        if self.outing is not None:
            forecast.recommendation = self.outing.predict(**weather)

        return forecast

    def predict_week(self, city: str, days_ahead: int = 7,
                     history_days: int = 10) -> WeeklyForecast:
        """指定した都市の、数日先までの天気とおすすめを返す。

        1日先（あした）は実測データから直接予測します。2日先以降は、
        前日ぶんの予測結果を「実測の続き」として入力に足し、もう一度
        同じモデルで予測する**再帰予測**で作ります。学習・特徴量づくりは
        1日先を当てる形のまま変えていないので、当てずっぽうの式を
        新しく作るのではなく、同じ検証済みの手順を繰り返し使えます。

        ただし、これは**予測を予測の材料にする**ということでもあります。
        1日目の誤差が2日目の入力に乗り、2日目の誤差が3日目に乗り……と、
        先に行くほど誤差が積み重なります。この積み重なりは検証していないので、
        3日目以降は「参考程度」と考えてください（doc/forecast.md 参照）。
        """
        days_ahead = max(1, min(days_ahead, MAX_FORECAST_DAYS))

        working = self.source.recent_daily(city, days=history_days)
        if working.empty:
            raise ForecastUnavailableError(
                f"{city} の直近の実測データが取得できませんでした。時間をおいて試してください。"
            )

        base_date = None
        days: list[DailyForecast] = []

        for day_index in range(1, days_ahead + 1):
            frame = feature_module.build_prediction_frame(working)
            if frame.empty:
                break  # これ以上は特徴量が作れない（通常は起きない）

            row = frame.iloc[[-1]]
            row_date = pd.to_datetime(row["date"].iloc[0])
            if base_date is None:
                base_date = row_date

            inputs = row[self.bundle.feature_names]
            weather = self._point_forecast(inputs)
            interval = self._widen_interval(self._interval_forecast(inputs), day_index)
            target_date = row_date + timedelta(days=1)

            recommendation = self.outing.predict(**weather) if self.outing else None

            days.append(
                DailyForecast(
                    day=day_index,
                    date=target_date.date().isoformat(),
                    weather=weather,
                    interval=interval,
                    recommendation=recommendation,
                )
            )

            # 予測した日を「実測の続き」として足し、次の日を予測する材料にする
            working = pd.concat(
                [
                    working,
                    pd.DataFrame([{**weather, "city": city, "date": target_date.date()}]),
                ],
                ignore_index=True,
            )

        if not days:
            raise ForecastUnavailableError(
                f"{city} の直近データが足りません（3日ぶん以上そろっている必要があります）。"
            )

        forecast = WeeklyForecast(
            city=city,
            base_date=base_date.date().isoformat(),
            days=days,
            model_version=self.bundle.version,
        )
        forecast.notes.append(
            "1日目より先は、予測した天気を入力に使い直して作っています（再帰予測）。"
            "誤差が日を追うごとに積み重なるため、3日目以降は参考程度に見てください。"
        )
        forecast.notes.append(
            "実測データは数日おくれて公開されるため、予測の起点は直近の入手可能日です。"
        )
        return forecast

    @staticmethod
    def _widen_interval(interval: dict[str, dict[str, float]],
                        day_index: int) -> dict[str, dict[str, float]]:
        """先の日ほど、予測区間を広げる（簡易な近似）。

        誤差が日ごとに独立に積み重なると仮定すると、ばらつきの大きさは
        だいたい日数の平方根に比例して増えます。厳密な統計的裏付けは
        取っていない、あくまで目安の広げ方です。
        """
        if day_index <= 1:
            return interval

        scale = day_index ** 0.5
        widened = {}
        for column, span in interval.items():
            center = (span["low"] + span["high"]) / 2
            half_width = (span["high"] - span["low"]) / 2 * scale

            low, high = INPUT_RANGES[column]
            widened[column] = {
                "low": round(float(np.clip(center - half_width, low, high)), 1),
                "high": round(float(np.clip(center + half_width, low, high)), 1),
            }
        return widened

    # -----------------------------------------------------------
    # 内部
    # -----------------------------------------------------------

    def _estimators(self):
        """成果物から、点予測と分位点のモデルを取り出す。"""
        estimator = self.bundle.estimator
        if not isinstance(estimator, dict) or "point" not in estimator:
            raise ForecastUnavailableError(
                "翌日予測モデルの形式が想定と違います。python train_forecast.py を実行し直してください。"
            )
        return estimator["point"], estimator.get("quantiles", {})

    def _point_forecast(self, inputs: pd.DataFrame) -> dict[str, float]:
        """あしたの天気4項目（真ん中の値）。"""
        point, _ = self._estimators()
        predicted = np.asarray(point.predict(inputs), dtype=float)[0]

        weather = {}
        for index, column in enumerate(FEATURE_COLUMNS):
            low, high = INPUT_RANGES[column]
            weather[column] = round(float(np.clip(predicted[index], low, high)), 1)
        return weather

    def _interval_forecast(self, inputs: pd.DataFrame) -> dict[str, dict[str, float]]:
        """あしたの天気の「だいたいこの範囲」。"""
        _, quantiles = self._estimators()
        if not quantiles:
            return {}

        low_quantile = min(CONFIG.forecast.quantiles)
        high_quantile = max(CONFIG.forecast.quantiles)

        interval = {}
        for column in FEATURE_COLUMNS:
            models = quantiles.get(column)
            if not models:
                continue

            lower = float(models[low_quantile].predict(inputs)[0])
            upper = float(models[high_quantile].predict(inputs)[0])
            # 分位点ごとに別々に学習しているので、上下が入れ替わることがある
            lower, upper = min(lower, upper), max(lower, upper)

            limit_low, limit_high = INPUT_RANGES[column]
            interval[column] = {
                "low": round(float(np.clip(lower, limit_low, limit_high)), 1),
                "high": round(float(np.clip(upper, limit_low, limit_high)), 1),
            }
        return interval

    def cities(self) -> list[str]:
        """予報を出せる都市の一覧。"""
        return self.source.city_names()

    def compare_tomorrow(self, cities: list[str] = None, days: int = 10) -> dict:
        """複数都市の、あしたの予報をまとめて返す。

        外部への通信を都市の数だけ行うため、連続で叩いて取得元の利用上限に
        当たらないよう、1都市ごとに間隔をあけます（fetch_weather.py の
        都市取得と同じ間隔を使う）。1都市ぶんの取得に失敗しても、
        残りの都市は続けます（失敗は errors にまとめる）。
        """
        cities = cities or self.cities()
        results = []
        errors = []

        for index, city in enumerate(cities):
            try:
                results.append(self.predict_tomorrow(city, days=days))
            except ForecastUnavailableError as error:
                errors.append({"city": city, "message": str(error)})

            if index < len(cities) - 1:
                time.sleep(CONFIG.data.request_interval_sec)

        return {"results": results, "errors": errors}
