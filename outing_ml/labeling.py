"""正解ラベルと正解スコアの作り方（弱教師あり学習）。

気象データには「その日どこへ行くべきか」「その日は何点か」という正解がありません。
そこで気象の指標をもとにしたルールで正解を作ります。

このやり方の弱点は正直に書いておきます。
モデルが学ぶのは「人の行動」ではなく「ここで決めたルール」です。
そのため精度が高くても、それは現実をよく当てているという意味ではなく、
決めた方針を忠実に再現できている、という意味になります。
実データ（施設の入場者数やアンケート）が手に入ったら、真っ先に置き換えるべき部分です。
"""


import numpy as np
import pandas as pd

from outing_ml.config import CATEGORIES, CONFIG, FEATURE_COLUMNS


def discomfort_index(temperature, humidity):
    """不快指数（蒸し暑さの指標）。日本の気象でよく使われる式。"""
    return 0.81 * temperature + 0.01 * humidity * (0.99 * temperature - 14.3) + 46.3


def felt_temperature(temperature, wind_speed):
    """体感温度。風があると同じ気温でも寒く感じる、という補正。"""
    return temperature - 1.5 * np.sqrt(wind_speed)


def outing_scores(df: pd.DataFrame, spec=None) -> np.ndarray:
    """1日ごとに、3カテゴリの「おすすめ度」を計算する（列の順は CATEGORIES）。"""
    spec = spec or CONFIG.label

    temperature = df["temperature"].to_numpy(dtype=float)
    rain = df["rain_probability"].to_numpy(dtype=float)
    wind = df["wind_speed"].to_numpy(dtype=float)
    humidity = df["humidity"].to_numpy(dtype=float)

    felt = felt_temperature(temperature, wind)
    discomfort = discomfort_index(temperature, humidity)

    outdoor = (
        spec.outdoor_peak
        * np.exp(
            -((temperature - spec.outdoor_best_temperature) ** 2)
            / (2 * spec.outdoor_temperature_width ** 2)
        )
        - spec.outdoor_rain_weight * rain
        - spec.outdoor_wind_weight * wind
        - spec.outdoor_humidity_weight
        * np.maximum(0.0, humidity - spec.outdoor_humidity_threshold)
    )

    indoor = (
        spec.indoor_base
        + spec.indoor_rain_weight * rain
        + spec.indoor_wind_weight * np.maximum(0.0, wind - spec.indoor_wind_threshold)
    )

    relax = (
        spec.relax_base
        + spec.relax_humidity_weight
        * np.maximum(0.0, humidity - spec.relax_humidity_threshold)
        + spec.relax_cold_weight * np.maximum(0.0, spec.relax_cold_threshold - felt)
        + spec.relax_hot_weight
        * np.maximum(0.0, temperature - spec.relax_hot_threshold)
        + spec.relax_discomfort_weight
        * np.maximum(0.0, discomfort - spec.relax_discomfort_threshold)
        - spec.relax_rain_weight * rain
    )

    columns = {"indoor": indoor, "outdoor": outdoor, "relax": relax}
    return np.column_stack([columns[name] for name in CATEGORIES])


def scores_to_probabilities(scores: np.ndarray, temperature: float = None) -> np.ndarray:
    """おすすめ度を、合計が1になる確率に変換する（ソフトマックス）。"""
    temperature = temperature or CONFIG.label.softmax_temperature
    scaled = scores / temperature
    exponent = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exponent / exponent.sum(axis=1, keepdims=True)


def label_probabilities(df: pd.DataFrame) -> np.ndarray:
    """天気から、3カテゴリの「本当の確率」を返す。"""
    return scores_to_probabilities(outing_scores(df))


def add_labels(df: pd.DataFrame, seed: int = None) -> pd.DataFrame:
    """正解ラベルの列と、その日の本当の確率の列を足す。

    確率にしたがってラベルを抽選するので、境目の日は答えがひとつに決まりません。
    これがこの問題の「越えられない壁」（ベイズ限界）になります。
    """
    seed = CONFIG.train.random_seed if seed is None else seed
    probabilities = label_probabilities(df)
    rng = np.random.default_rng(seed)

    draws = rng.random(len(df))
    cumulative = probabilities.cumsum(axis=1)
    chosen = (draws[:, None] > cumulative).sum(axis=1)
    chosen = np.clip(chosen, 0, len(CATEGORIES) - 1)

    result = df.copy()
    result["label"] = [CATEGORIES[index] for index in chosen]
    for index, name in enumerate(CATEGORIES):
        result[f"true_prob_{name}"] = probabilities[:, index]
    return result


def bayes_accuracy(df: pd.DataFrame) -> float:
    """理論上の正解率の上限。いちばん確率の高いカテゴリを答え続けたときの正解率。"""
    columns = [f"true_prob_{name}" for name in CATEGORIES]
    if all(column in df.columns for column in columns):
        probabilities = df[columns].to_numpy()
    else:
        probabilities = label_probabilities(df)
    return float(probabilities.max(axis=1).mean())


# ---------------------------------------------------------------
# おでかけ日和度（0〜100点）
# ---------------------------------------------------------------

def true_comfort_score(df: pd.DataFrame, spec=None) -> np.ndarray:
    """天気から、おでかけ日和度の素点（0〜100）を減点方式で計算する。"""
    spec = spec or CONFIG.comfort

    temperature = df["temperature"].to_numpy(dtype=float)
    rain = df["rain_probability"].to_numpy(dtype=float)
    wind = df["wind_speed"].to_numpy(dtype=float)
    humidity = df["humidity"].to_numpy(dtype=float)

    felt = felt_temperature(temperature, wind)
    discomfort = discomfort_index(temperature, humidity)

    penalty = (
        spec.cold_weight * np.maximum(0.0, spec.cold_threshold - felt)
        + spec.hot_weight * np.maximum(0.0, temperature - spec.hot_threshold)
        + spec.rain_weight * rain
        + spec.wind_weight * np.maximum(0.0, wind - spec.wind_threshold)
        + spec.discomfort_weight
        * np.maximum(0.0, discomfort - spec.discomfort_threshold)
    )

    return np.clip(100.0 - penalty, 0.0, 100.0)


def add_comfort_scores(df: pd.DataFrame, seed: int = None) -> pd.DataFrame:
    """素点と、人による感じ方のちがいを足した正解の点数を足す。"""
    seed = CONFIG.train.random_seed if seed is None else seed
    rng = np.random.default_rng(seed)

    true_score = true_comfort_score(df)
    noise = rng.normal(0.0, CONFIG.comfort.noise_std, len(df))

    result = df.copy()
    result["true_score"] = true_score
    result["comfort_score"] = np.clip(true_score + noise, 0.0, 100.0)
    return result


def label_distribution(df: pd.DataFrame) -> dict[str, int]:
    """ラベルの件数を数える（カテゴリの並び順を固定して返す）。"""
    counts = df["label"].value_counts()
    return {name: int(counts.get(name, 0)) for name in CATEGORIES}


def check_feature_columns(df: pd.DataFrame) -> None:
    """特徴量の列がそろっているか確かめる。"""
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"特徴量の列がありません: {missing}")
