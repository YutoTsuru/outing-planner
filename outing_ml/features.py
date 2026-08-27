"""翌日予測のための特徴量づくり。

時系列で怖いのは「未来の情報が、こっそり入力にまざる」ことです（データ漏洩）。
そうなると検証の点数だけが良くなり、本番でまったく当たらないモデルができます。

この モジュールでは、
  ・入力は「その日まで」の値だけ
  ・正解は「その次の日」の値
と決め、テスト（tests/test_features.py）でも守られていることを確かめています。
"""


import numpy as np
import pandas as pd

from outing_ml.config import CONFIG, FEATURE_COLUMNS

# 正解の列につける印
TARGET_SUFFIX = "_next"


def _build_lag_frame(raw: pd.DataFrame, spec, with_target: bool) -> pd.DataFrame:
    """ラグ・移動平均・前日差・季節をまとめて作る（学習用と予測用の共通部分）。

    学習と本番で作り方が1文字でも違うと、静かに精度が落ちます。
    with_target を切り替えるだけにして、特徴量の作り方は必ず1か所に保ちます。
    """
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

        if with_target:
            # 正解（あしたの値）
            features[f"{column}{TARGET_SUFFIX}"] = grouped[column].shift(-1)

    # 季節。1年でひと回りするので、角度に直して sin と cos の2つで表す
    day_of_year = df["date"].dt.dayofyear
    features["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    return features


def build_supervised_frame(raw: pd.DataFrame, spec=None) -> pd.DataFrame:
    """1日ずつの気象データを、「あしたを当てる」ための表に組み直す（学習用）。"""
    spec = spec or CONFIG.forecast
    return _build_lag_frame(raw, spec, with_target=True).dropna().reset_index(drop=True)


def build_prediction_frame(raw: pd.DataFrame, spec=None) -> pd.DataFrame:
    """いちばん新しい日から「あした」を予測するための1行を、都市ごとに作る（本番用）。

    学習用と違って正解の列は作りません。作ってしまうと、
    正解が無い最新の日（＝まさに予測したい日）が dropna で落ちてしまいます。
    """
    spec = spec or CONFIG.forecast
    features = _build_lag_frame(raw, spec, with_target=False)
    features = features.dropna().reset_index(drop=True)

    if features.empty:
        return features

    # 都市ごとに、いちばん新しい日だけを残す
    newest = features.groupby("city")["date"].transform("max")
    return features[features["date"] == newest].reset_index(drop=True)


def input_columns(features: pd.DataFrame) -> list[str]:
    """入力に使う列の名前（正解の列と日付をのぞいたもの）。"""
    return [
        column
        for column in features.columns
        if column != "date" and not column.endswith(TARGET_SUFFIX)
    ]


def target_columns() -> list[str]:
    """正解の列の名前。"""
    return [f"{column}{TARGET_SUFFIX}" for column in FEATURE_COLUMNS]


def split_by_year(features: pd.DataFrame, test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
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
