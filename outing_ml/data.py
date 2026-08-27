"""データの読み込みと検証。

実務では「モデルが壊れた」より「データが壊れた」ほうがはるかに多く起きます。
学習でも推論でも、データは必ずここを通して検証します。
おかしなデータは、静かに悪い予測を出すのではなく、その場で止めます。
"""

import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from outing_ml.config import CONFIG, FEATURE_COLUMNS


class DataValidationError(ValueError):
    """データが約束（スキーマ）を満たしていないときに投げる例外。"""


@dataclass(frozen=True)
class ColumnSpec:
    """1列ぶんの約束ごと。"""

    name: str
    kind: str                      # "number" / "text" / "date"
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    required: bool = True


# 学習データ（data/weather_jp.csv）の約束
# 範囲は「物理的にありえる値」で、アプリの入力範囲より広く取っている
DATASET_SCHEMA: List[ColumnSpec] = [
    ColumnSpec("city", "text"),
    ColumnSpec("date", "date"),
    ColumnSpec("temperature", "number", minimum=-40.0, maximum=50.0),
    ColumnSpec("rain_probability", "number", minimum=0.0, maximum=100.0),
    ColumnSpec("wind_speed", "number", minimum=0.0, maximum=60.0),
    ColumnSpec("humidity", "number", minimum=0.0, maximum=100.0),
]


@dataclass
class ValidationReport:
    """検証の結果。errors が空でなければ、そのデータは使えない。"""

    rows: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> "ValidationReport":
        if not self.ok:
            raise DataValidationError(
                "データの検証に失敗しました:\n  - " + "\n  - ".join(self.errors)
            )
        return self


def validate_frame(df: pd.DataFrame, schema: List[ColumnSpec] = None) -> ValidationReport:
    """DataFrame がスキーマを満たしているか調べる。

    調べること:
      1. 必要な列がそろっているか
      2. 数値の列がちゃんと数値か
      3. 値が物理的にありえる範囲か
      4. 欠損（空欄）が無いか
      5. 同じ都市・同じ日が二重に入っていないか
      6. 日付がとびとびになっていないか（時系列の学習で効いてくる）
    """
    schema = schema or DATASET_SCHEMA
    report = ValidationReport(rows=len(df))

    if len(df) == 0:
        report.errors.append("行が1件もありません")
        return report

    for column in schema:
        if column.name not in df.columns:
            if column.required:
                report.errors.append(f"列 '{column.name}' がありません")
            continue

        values = df[column.name]

        missing = int(values.isna().sum())
        if missing:
            report.errors.append(f"列 '{column.name}' に欠損が {missing} 件あります")

        if column.kind == "number":
            numeric = pd.to_numeric(values, errors="coerce")
            broken = int(numeric.isna().sum() - missing)
            if broken > 0:
                report.errors.append(
                    f"列 '{column.name}' に数値でない値が {broken} 件あります"
                )

            valid = numeric.dropna()
            if len(valid):
                report.stats[column.name] = {
                    "min": float(valid.min()),
                    "max": float(valid.max()),
                    "mean": float(valid.mean()),
                    "std": float(valid.std()),
                }
                if column.minimum is not None and valid.min() < column.minimum:
                    report.errors.append(
                        f"列 '{column.name}' に下限 {column.minimum} を下回る値があります"
                        f"（最小 {valid.min()}）"
                    )
                if column.maximum is not None and valid.max() > column.maximum:
                    report.errors.append(
                        f"列 '{column.name}' に上限 {column.maximum} を上回る値があります"
                        f"（最大 {valid.max()}）"
                    )

        if column.kind == "date":
            parsed = pd.to_datetime(values, errors="coerce")
            broken = int(parsed.isna().sum() - missing)
            if broken > 0:
                report.errors.append(
                    f"列 '{column.name}' に日付として読めない値が {broken} 件あります"
                )

    if "city" in df.columns and "date" in df.columns:
        duplicated = int(df.duplicated(["city", "date"]).sum())
        if duplicated:
            report.errors.append(f"同じ都市・同じ日の行が {duplicated} 件あります")

        gaps = _find_date_gaps(df)
        if gaps:
            report.warnings.append(
                "日付がとびとびになっている都市があります: " + "、".join(gaps[:5])
            )

    return report


def _find_date_gaps(df: pd.DataFrame) -> List[str]:
    """都市ごとに、日付が1日ずつ連続しているかを調べる。"""
    gaps = []
    frame = df[["city", "date"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    for city, group in frame.dropna().groupby("city"):
        dates = group["date"].sort_values()
        expected = (dates.iloc[-1] - dates.iloc[0]).days + 1
        if len(dates) != expected:
            gaps.append(f"{city}（{len(dates)}日 / 期待 {expected}日）")

    return gaps


def load_dataset(path: str = None, validate: bool = True) -> pd.DataFrame:
    """学習データを読み込む（既定で検証つき）。"""
    path = path or CONFIG.paths.dataset

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"データが見つかりません: {path}\n"
            "先に python fetch_weather.py を実行してください。"
        )

    df = pd.read_csv(path)
    if validate:
        validate_frame(df).raise_if_failed()
    return df


def file_sha256(path: str) -> str:
    """ファイルの中身から指紋（ハッシュ）を作る。

    「どのデータで学習したモデルなのか」を後から証明できるようにするため、
    学習した成果物に必ずこの値を残します。
    """
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(path: str = None) -> Dict[str, object]:
    """学習データの指紋（ハッシュ・行数・期間）をまとめて返す。"""
    path = path or CONFIG.paths.dataset
    df = pd.read_csv(path)

    return {
        "path": path,
        "sha256": file_sha256(path),
        "rows": int(len(df)),
        "cities": int(df["city"].nunique()) if "city" in df.columns else None,
        "date_from": str(df["date"].min()) if "date" in df.columns else None,
        "date_to": str(df["date"].max()) if "date" in df.columns else None,
    }


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量の4列だけを、決まった順番で取り出す。"""
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise DataValidationError(f"特徴量の列がありません: {missing}")
    return df[FEATURE_COLUMNS]
