"""推論の入口（サービング）。

アプリから直接 joblib.load して predict すると、次の事故が起きます。

  ・特徴量の順番がズレても、エラーにならず静かに間違える
  ・学習していない範囲の値を渡しても、それと分からず答えを返す
  ・モデルを差し替えたとき、アプリ側の修正もれに気づけない

そこで、推論はすべてこのモジュールを通します。
入力を検証し、学習時の範囲から外れていれば警告をつけて返します。
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from outing_ml.config import CATEGORIES, CONFIG, FEATURE_COLUMNS, INPUT_RANGES
from outing_ml.registry import ModelBundle, load_bundle


class InvalidInputError(ValueError):
    """入力が数値として扱えないときに投げる例外。"""


@dataclass
class Recommendation:
    """1日ぶんの予測結果。"""

    category: str
    probabilities: dict[str, float]
    confidence: float
    comfort_score: float | None = None
    weather_type: int | None = None
    weather_type_name: str | None = None
    warnings: list[str] = field(default_factory=list)
    model_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "comfort_score": self.comfort_score,
            "weather_type": self.weather_type,
            "weather_type_name": self.weather_type_name,
            "warnings": self.warnings,
            "model_versions": self.model_versions,
        }


def _as_number(name: str, value) -> float:
    """入力を数値に直す。数値にできなければ、その場で止める。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise InvalidInputError(
            f"{name} は数値で指定してください（受け取った値: {value!r}）"
        ) from error

    if not np.isfinite(number):
        raise InvalidInputError(f"{name} に数値でない値（NaN / inf）が渡されました")

    return number


def validate_weather(values: dict[str, float]) -> (dict[str, float], list[str]):
    """入力の4項目を検証して、（そろえた値, 警告）を返す。

    アプリで入力できる範囲を外れた値は、止めずに範囲内へ丸めます。
    予測を返さないより、丸めたうえで警告を出すほうが、使う側が困りません。
    """
    cleaned = {}
    warnings = []

    for name in FEATURE_COLUMNS:
        if name not in values:
            raise InvalidInputError(f"{name} が指定されていません")

        number = _as_number(name, values[name])
        low, high = INPUT_RANGES[name]

        if number < low or number > high:
            warnings.append(
                f"{name} が入力できる範囲（{low}〜{high}）の外だったため、"
                f"{max(low, min(high, number))} として扱いました"
            )
            number = max(low, min(high, number))

        cleaned[name] = number

    return cleaned, warnings


class OutingService:
    """4つのモデルをまとめて扱う、推論用のサービス。

    カテゴリ予測モデルは必須です。
    それ以外（日和度・天気タイプ）は無くても動きます（その項目が None になるだけ）。
    """

    def __init__(self, category: ModelBundle, comfort: ModelBundle = None,
                 weather_type: ModelBundle = None):
        self.category = category
        self.comfort = comfort
        self.weather_type = weather_type

    @classmethod
    def load(cls, paths=None, strict: bool = False) -> "OutingService":
        """成果物を読み込んでサービスを作る。"""
        paths = paths or CONFIG.paths
        category = load_bundle(paths.category_model)

        optional = {}
        for name, path in (("comfort", paths.comfort_model),
                           ("weather_type", paths.weather_type_model)):
            try:
                optional[name] = load_bundle(path)
            except FileNotFoundError:
                if strict:
                    raise
                optional[name] = None

        return cls(category=category, **optional)

    # -----------------------------------------------------------
    # 予測
    # -----------------------------------------------------------

    def predict(self, temperature, rain_probability, wind_speed, humidity) -> Recommendation:
        """1日ぶんの天気から、おすすめを返す。"""
        cleaned, warnings = validate_weather(
            {
                "temperature": temperature,
                "rain_probability": rain_probability,
                "wind_speed": wind_speed,
                "humidity": humidity,
            }
        )

        frame = pd.DataFrame([cleaned], columns=FEATURE_COLUMNS)
        warnings.extend(self._training_range_warnings(cleaned))

        self.category.check_features(FEATURE_COLUMNS)
        probabilities = self.category.estimator.predict_proba(frame)[0]
        classes = list(self.category.classes or self.category.estimator.classes_)
        best = int(np.argmax(probabilities))

        result = Recommendation(
            category=classes[best],
            probabilities={
                name: float(round(value, 4))
                for name, value in zip(classes, probabilities, strict=True)
            },
            confidence=float(round(probabilities[best], 4)),
            warnings=warnings,
            model_versions={"category": self.category.version},
        )

        if self.comfort is not None:
            score = float(self.comfort.estimator.predict(frame)[0])
            result.comfort_score = round(min(100.0, max(0.0, score)), 1)
            result.model_versions["comfort"] = self.comfort.version

        if self.weather_type is not None:
            type_id = int(self.weather_type.estimator.predict(frame)[0])
            result.weather_type = type_id
            names = self.weather_type.metadata.get("cluster_names", {})
            result.weather_type_name = names.get(str(type_id)) or names.get(type_id)
            result.model_versions["weather_type"] = self.weather_type.version

        return result

    def predict_batch(self, frame: pd.DataFrame) -> pd.DataFrame:
        """まとめて予測する（監視やバックテストで使う）。"""
        missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
        if missing:
            raise InvalidInputError(f"列がありません: {missing}")

        features = frame[FEATURE_COLUMNS].astype(float)
        probabilities = self.category.estimator.predict_proba(features)
        classes = list(self.category.classes or self.category.estimator.classes_)

        result = pd.DataFrame(
            probabilities, columns=[f"proba_{name}" for name in classes], index=frame.index
        )
        result["category"] = [classes[index] for index in probabilities.argmax(axis=1)]
        result["confidence"] = probabilities.max(axis=1)

        if self.comfort is not None:
            result["comfort_score"] = np.clip(
                self.comfort.estimator.predict(features), 0.0, 100.0
            )
        if self.weather_type is not None:
            result["weather_type"] = self.weather_type.estimator.predict(features)

        return result

    # -----------------------------------------------------------
    # 補助
    # -----------------------------------------------------------

    def _training_range_warnings(self, values: dict[str, float]) -> list[str]:
        """学習データに無かった範囲の値には、警告をつける。

        予測は返しますが、その根拠は弱い（外挿している）ことを伝えます。
        """
        ranges = self.category.metadata.get("feature_ranges") or {}
        warnings = []

        for name, value in values.items():
            limits = ranges.get(name)
            if not limits:
                continue
            if value < limits["min"] or value > limits["max"]:
                warnings.append(
                    f"{name}={value} は学習データの範囲"
                    f"（{limits['min']}〜{limits['max']}）の外です。予測の根拠は弱くなります"
                )

        return warnings

    def health(self) -> dict[str, object]:
        """読み込めているモデルと、その版を返す（起動確認用）。"""
        return {
            "ok": True,
            "categories": list(self.category.classes or CATEGORIES),
            "features": list(self.category.feature_names),
            "models": {
                "category": self.category.version,
                "comfort": self.comfort.version if self.comfort else None,
                "weather_type": self.weather_type.version if self.weather_type else None,
            },
            "trained_at": self.category.metadata.get("created_at"),
            "data_sha256": (self.category.metadata.get("data") or {}).get("sha256"),
        }
