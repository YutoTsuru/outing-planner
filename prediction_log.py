"""予測の記録の読み書き。

いまの正解ラベルはルールから作った疑似データです（doc/README.md 第4章）。
実際に使われた天気を貯めておけば、いずれ本物の利用データでモデルを作り直せます。

記録するのは天気の数値と予測結果だけで、個人を特定できる情報は残しません。
記録に失敗しても予測は返します（記録は本筋ではないため）。
"""

import json
import os
from collections import Counter, deque
from datetime import UTC, datetime

import pandas as pd

from outing_ml.config import CATEGORIES, CONFIG, FEATURE_COLUMNS

# 1行1件の JSON Lines。追記しかしない
LOG_PATH = os.path.join(CONFIG.paths.report_dir, "predictions.jsonl")

# 記録を切りたいときは OUTING_LOG_PREDICTIONS=0
ENV_FLAG = "OUTING_LOG_PREDICTIONS"

# 画面や API で一度に返す既定の件数
DEFAULT_LIMIT = 100

# 読み込む行数の上限。増え続けても画面が重くならないようにする
MAX_SCAN_LINES = 20000


def enabled() -> bool:
    """記録を取る設定になっているか。"""
    return os.environ.get(ENV_FLAG, "1") != "0"


def append(kind: str, payload: dict, path: str = None) -> None:
    """1件ぶんを追記する（best-effort。失敗しても例外にしない）。"""
    if not enabled():
        return

    path = path or LOG_PATH
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        line = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "kind": kind,
            **payload,
        }
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError:
        pass


def read_entries(limit: int = DEFAULT_LIMIT, kinds=None, path: str = None) -> list[dict]:
    """新しい順に読み出す。

    途中で書き込みが切れた行（JSON として読めない行）は飛ばします。
    追記中のファイルを読むので、最後の1行が欠けていることがあるためです。
    """
    path = path or LOG_PATH
    if not os.path.exists(path):
        return []

    kept = deque(maxlen=MAX_SCAN_LINES)
    try:
        with open(path, encoding="utf-8") as file:
            for raw in file:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if kinds and entry.get("kind") not in kinds:
                    continue
                kept.append(entry)
    except OSError:
        return []

    entries = list(kept)
    entries.reverse()                 # 新しい順
    return entries[:limit] if limit else entries


def summarize(entries: list[dict]) -> dict:
    """記録の傾向をまとめる。

    「どんな天気のときに使われたか」が分かると、学習データの範囲と
    実際の使われ方がずれていないかを目で確かめられます。
    """
    if not entries:
        return {"total": 0, "by_kind": {}, "by_category": [],
                "period": None, "averages": {}, "average_confidence": None}

    kinds = Counter(entry.get("kind", "unknown") for entry in entries)
    categories = Counter(
        entry["category"] for entry in entries if entry.get("category")
    )
    total_with_category = sum(categories.values())

    sums = dict.fromkeys(FEATURE_COLUMNS, 0.0)
    counts = dict.fromkeys(FEATURE_COLUMNS, 0)
    for entry in entries:
        weather = entry.get("input") or entry.get("weather") or {}
        for column in FEATURE_COLUMNS:
            value = weather.get(column)
            if isinstance(value, (int, float)):
                sums[column] += float(value)
                counts[column] += 1

    confidences = [
        entry["confidence"] for entry in entries
        if isinstance(entry.get("confidence"), (int, float))
    ]

    timestamps = sorted(entry["at"] for entry in entries if entry.get("at"))

    return {
        "total": len(entries),
        "by_kind": dict(kinds.most_common()),
        "by_category": [
            {
                "category": name,
                "count": categories.get(name, 0),
                "share": round(categories.get(name, 0) / total_with_category, 4)
                if total_with_category else 0.0,
            }
            for name in CATEGORIES
        ],
        "period": {"from": timestamps[0], "to": timestamps[-1]} if timestamps else None,
        "averages": {
            column: round(sums[column] / counts[column], 1)
            for column in FEATURE_COLUMNS if counts[column]
        },
        "average_confidence": round(sum(confidences) / len(confidences), 4)
        if confidences else None,
    }


def weather_frame(entries: list[dict]) -> pd.DataFrame:
    """記録から、天気4項目だけの表を作る（ドリフト監視で使う）。

    predict 系の記録は "input"、forecast 系の記録は "weather" というキーに
    天気が入っている。どちらも同じ4項目なので、ここで1つの表にそろえる。
    """
    rows = []
    for entry in entries:
        weather = entry.get("input") or entry.get("weather")
        if not weather:
            continue
        if all(isinstance(weather.get(column), (int, float)) for column in FEATURE_COLUMNS):
            rows.append({column: float(weather[column]) for column in FEATURE_COLUMNS})

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
