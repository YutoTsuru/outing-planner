"""設定の一元管理。

しきい値やハイパーパラメータをコードのあちこちに書くと、
学習時と推論時で値がズレて事故になります（train/serve skew）。
このプロジェクトでは、そういった値をすべてこのファイルに集めています。
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------
# 列と分類ラベル（学習・推論・監視で共通）
# ---------------------------------------------------------------

# 特徴量の順番。この順番はモデルの成果物にも保存され、推論時に検証される
FEATURE_COLUMNS: list[str] = [
    "temperature",
    "rain_probability",
    "wind_speed",
    "humidity",
]

# 予測するカテゴリ（確率の並び順もこの通り）
CATEGORIES: list[str] = ["indoor", "outdoor", "relax"]

# アプリで入力できる範囲（＝推論時に受け付ける範囲）
INPUT_RANGES: dict[str, tuple[float, float]] = {
    "temperature": (-10.0, 40.0),
    "rain_probability": (0.0, 100.0),
    "wind_speed": (0.0, 20.0),
    "humidity": (0.0, 100.0),
}


# ---------------------------------------------------------------
# ファイルの置き場所
# ---------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    data_dir: str = "data"
    model_dir: str = "model"
    report_dir: str = "reports"

    # 学習に使うデータ（2019〜2024年）
    dataset: str = "data/weather_jp.csv"
    # 学習後に一度だけ使う「未来のデータ」（2025年以降）
    holdout: str = "data/weather_jp_holdout.csv"

    registry: str = "model/registry.json"

    category_model: str = "model/outing_model.pkl"
    forecast_model: str = "model/forecast_model.pkl"
    comfort_model: str = "model/comfort_model.pkl"
    weather_type_model: str = "model/weather_type_model.pkl"

    category_card: str = "model/model_card.json"
    forecast_card: str = "model/forecast_card.json"
    comfort_card: str = "model/comfort_card.json"
    weather_type_card: str = "model/weather_types.json"


# ---------------------------------------------------------------
# データの取得条件
# ---------------------------------------------------------------

@dataclass(frozen=True)
class DataSpec:
    # (都市名, 緯度, 経度)。全都道府県の県庁所在地。
    #
    # 9都市から47都市へ広げた理由: モデルは都市名を入力に使うため、学習していない
    # 都市を渡すと当てずっぽうになる。全県を入れておけば、日本のどこでも使える。
    cities: tuple[tuple[str, float, float], ...] = (
        ("札幌", 43.0621, 141.3544), ("青森", 40.8244, 140.7400),
        ("盛岡", 39.7036, 141.1527), ("仙台", 38.2682, 140.8694),
        ("秋田", 39.7186, 140.1024), ("山形", 38.2404, 140.3633),
        ("福島", 37.7503, 140.4676), ("水戸", 36.3418, 140.4468),
        ("宇都宮", 36.5657, 139.8836), ("前橋", 36.3912, 139.0608),
        ("さいたま", 35.8569, 139.6489), ("千葉", 35.6051, 140.1233),
        ("東京", 35.6895, 139.6917), ("横浜", 35.4478, 139.6425),
        ("新潟", 37.9161, 139.0364), ("富山", 36.6953, 137.2114),
        ("金沢", 36.5947, 136.6256), ("福井", 36.0652, 136.2216),
        ("甲府", 35.6642, 138.5686), ("長野", 36.6513, 138.1810),
        ("岐阜", 35.3912, 136.7223), ("静岡", 34.9769, 138.3831),
        ("名古屋", 35.1815, 136.9066), ("津", 34.7303, 136.5086),
        ("大津", 35.0045, 135.8686), ("京都", 35.0116, 135.7681),
        ("大阪", 34.6937, 135.5023), ("神戸", 34.6913, 135.1830),
        ("奈良", 34.6851, 135.8048), ("和歌山", 34.2261, 135.1675),
        ("鳥取", 35.5039, 134.2377), ("松江", 35.4723, 133.0505),
        ("岡山", 34.6618, 133.9350), ("広島", 34.3853, 132.4553),
        ("山口", 34.1859, 131.4706), ("徳島", 34.0658, 134.5593),
        ("高松", 34.3401, 134.0434), ("松山", 33.8416, 132.7657),
        ("高知", 33.5597, 133.5311), ("福岡", 33.5904, 130.4017),
        ("佐賀", 33.2494, 130.2989), ("長崎", 32.7448, 129.8737),
        ("熊本", 32.7898, 130.7417), ("大分", 33.2382, 131.6126),
        ("宮崎", 31.9111, 131.4239), ("鹿児島", 31.5602, 130.5581),
        ("那覇", 26.2124, 127.6809),
    )
    start_date: str = "2019-01-01"
    end_date: str = "2024-12-31"
    # 未来データ（ホールドアウト）の開始日。終わりは実行日の1週間前
    holdout_start_date: str = "2025-01-01"
    holdout_lag_days: int = 7

    # お出かけする時間帯（9時〜18時の10時間）
    outing_hours: tuple[int, ...] = tuple(range(9, 19))
    # 「雨が降っている」とみなす1時間あたりの雨量（mm）
    rain_threshold_mm: float = 0.1
    request_interval_sec: float = 1.0
    request_timeout_sec: int = 120


# ---------------------------------------------------------------
# ラベル付け（おすすめ度モデル）
# ---------------------------------------------------------------

@dataclass(frozen=True)
class LabelSpec:
    """天気から3カテゴリの「おすすめ度」を出すときの重み。

    ここを変えるとラベルの意味が変わるため、変更したら必ず全モデルを学習し直します。
    """

    softmax_temperature: float = 0.6

    # 屋外
    outdoor_peak: float = 3.6
    outdoor_best_temperature: float = 22.0
    outdoor_temperature_width: float = 8.5
    outdoor_rain_weight: float = 0.048
    outdoor_wind_weight: float = 0.200
    outdoor_humidity_weight: float = 0.020
    outdoor_humidity_threshold: float = 70.0

    # 屋内
    indoor_base: float = 0.35
    indoor_rain_weight: float = 0.040
    indoor_wind_weight: float = 0.220
    indoor_wind_threshold: float = 4.0

    # リラックス
    relax_base: float = 0.80
    relax_humidity_weight: float = 0.050
    relax_humidity_threshold: float = 72.0
    relax_cold_weight: float = 0.150
    relax_cold_threshold: float = 13.0
    relax_hot_weight: float = 0.160
    relax_hot_threshold: float = 27.0
    relax_discomfort_weight: float = 0.025
    relax_discomfort_threshold: float = 76.0
    relax_rain_weight: float = 0.008


@dataclass(frozen=True)
class ComfortSpec:
    """おでかけ日和度（0〜100点）の減点の重み。"""

    cold_weight: float = 1.8
    cold_threshold: float = 22.0
    hot_weight: float = 2.4
    hot_threshold: float = 26.0
    rain_weight: float = 0.45
    wind_weight: float = 1.6
    wind_threshold: float = 3.0
    discomfort_weight: float = 1.2
    discomfort_threshold: float = 75.0
    # 人による感じ方のちがい（点数のばらつき）
    noise_std: float = 6.0


# ---------------------------------------------------------------
# 学習の条件
# ---------------------------------------------------------------

@dataclass(frozen=True)
class TrainSpec:
    random_seed: int = 42
    test_size: float = 0.2
    cv_splits: int = 5
    bootstrap_samples: int = 1000
    calibration_bins: int = 10


@dataclass(frozen=True)
class ForecastSpec:
    lag_days: tuple[int, ...] = (1, 2)
    rolling_window: int = 3
    # ウォークフォワード検証の分割（学習の終わり, 評価する年）
    backtest_years: tuple[int, ...] = (2021, 2022, 2023, 2024)
    # 予測区間の分位点
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class MonitorSpec:
    """ドリフト監視のしきい値（金融でよく使われる目安に合わせている）。"""

    psi_bins: int = 10
    psi_watch: float = 0.10   # これを超えたら要観察
    psi_alert: float = 0.25   # これを超えたら再学習を検討
    ks_alert: float = 0.20


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    data: DataSpec = field(default_factory=DataSpec)
    label: LabelSpec = field(default_factory=LabelSpec)
    comfort: ComfortSpec = field(default_factory=ComfortSpec)
    train: TrainSpec = field(default_factory=TrainSpec)
    forecast: ForecastSpec = field(default_factory=ForecastSpec)
    monitor: MonitorSpec = field(default_factory=MonitorSpec)


CONFIG = Config()
