"""翌日予測のための特徴量づくり。

時系列で怖いのは「未来の情報が、こっそり入力にまざる」ことです（データ漏洩）。
そうなると検証の点数だけが良くなり、本番でまったく当たらないモデルができます。

この モジュールでは、
  ・入力は「その日まで」の値だけ
  ・正解は「その次の日」の値
と決め、テスト（tests/test_features.py）でも守られていることを確かめています。
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from outing_ml.config import CONFIG, FEATURE_COLUMNS

# 正解の列につける印
TARGET_SUFFIX = "_next"


def build_supervised_frame(raw: pd.DataFrame, spec=None) -> pd.DataFrame:
    """1日ずつの気象データを、「あしたを当てる」ための表に組み直す。"""
    spec = spec or CONFIG.forecast

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    grouped = df.groupby("city", sort=False)
    features = pd.DataFrame({"city": df["city"], "date": df["date"]})

    for column in FEATURE_COLUMNS:
        features[f"{column}_today"] = df[column]

        for lag in spec.lag_days:
            features[f"{column}_lag{lag}"] = grouped[column].shift(lag)

        features[f"{column}_mean{spec.rolling_window}"] = (
            grouped[column]
            .rolling(spec.rolling_window, min_periods=spec.rolling_window)
            .mean()
            .reset_index(level=0, drop=True)
        )
        features[f"{column}_diff"] = df[column] - grouped[column].shift(1)

        # 正解（あしたの値）
        features[f"{column}{TARGET_SUFFIX}"] = grouped[column].shift(-1)

    # 季節。1年でひと回りするので、角度に直して sin と cos の2つで表す
    day_of_year = df["date"].dt.dayofyear
    features["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    return features.dropna().reset_index(drop=True)


def input_columns(features: pd.DataFrame) -> List[str]:
    """入力に使う列の名前（正解の列と日付をのぞいたもの）。"""
    return [
        column
        for column in features.columns
        if column != "date" and not column.endswith(TARGET_SUFFIX)
    ]


def target_columns() -> List[str]:
    """正解の列の名前。"""
    return [f"{column}{TARGET_SUFFIX}" for column in FEATURE_COLUMNS]


def split_by_year(features: pd.DataFrame, test_year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """指定した年より前を学習用、その年以降を評価用に分ける。"""
    is_test = features["date"].dt.year >= test_year
    return (
        features[~is_test].reset_index(drop=True),
        features[is_test].reset_index(drop=True),
    )


def walk_forward_splits(features: pd.DataFrame, years=None):
    """ウォークフォワード検証の分割を作る。

    実務の時系列モデルは、1回の分割だけで良し悪しを決めません。
    「2020年までで学習して2021年を予測」「2021年までで学習して2022年を予測」…と
    実際の運用と同じ順番で、何度も検証します。
    """
    years = years or CONFIG.forecast.backtest_years

    for year in years:
        train = features[features["date"].dt.year < year]
        test = features[features["date"].dt.year == year]
        if len(train) == 0 or len(test) == 0:
            continue
        yield year, train.reset_index(drop=True), test.reset_index(drop=True)


def assert_no_leakage(features: pd.DataFrame) -> None:
    """入力に未来の情報がまざっていないかを確かめる。

    「あしたの値（_next）」が入力側の列に入っていたら、その場で止めます。
    """
    leaked = [column for column in input_columns(features) if column.endswith(TARGET_SUFFIX)]
    if leaked:
        raise ValueError(f"未来の情報が入力にまざっています: {leaked}")
