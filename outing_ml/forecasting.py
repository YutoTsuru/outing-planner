"""翌日の天気予報と、そこから出すおすすめ。

学習した2つのモデルをつなげて使います。

    直近の実測（Open-Meteo） → 翌日予測モデル → あしたの天気
                                                → カテゴリ予測モデル → あしたのおすすめ

予報の誤差はそのままおすすめのズレになります（実測 2025〜2026 年で一致率 61.8%）。
そのため、予測した天気には必ず予測区間（だいたいこの範囲）を添えて返します。
"""

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
