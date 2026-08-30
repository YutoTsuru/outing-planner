"""アプリのドリフト監視。

outing_ml.monitor は「基準データ」と「比べたいデータ」を渡せば
ずれを測ってくれる汎用の道具です。このモジュールはその一歩手前、
「アプリの実際の使われ方に合わせて、どのデータを基準・比較に使うか」を決めます。

理想は、学習データ（基準）と、実際にユーザーが入力した天気（比較）を比べることです。
ただし使いはじめのアプリでは記録がまだ少なく、それでは判定が安定しません。
そこで記録が少ないうちは、学習後に一度も見ていない「未来データ」を代わりに使い、
その旨を報告に残します。
"""

import os

import pandas as pd

import prediction_log
from outing_ml import monitor
from outing_ml.config import CONFIG
from outing_ml.data import load_dataset
from outing_ml.registry import load_bundle
from outing_ml.serve import OutingService

# 実際の入力を使うために必要な最少件数。
# これより少ないと、PSI（10個の区間に分ける）が安定して計算できない。
MIN_LIVE_ROWS = 30

# 「最近の入力」として見る記録の種類（天気の数値を持つもの）
LIVE_KINDS = {"predict", "predict_batch", "forecast", "web_predict", "web_forecast"}

# 比較に使う記録は、直近どれだけ遡るか
LIVE_LOOKBACK = 5000


class MonitorUnavailableError(RuntimeError):
    """比較に使えるデータが1つも無いときに投げる例外。"""


def _live_current_frame() -> pd.DataFrame:
    """実際の入力から、比較用の天気の表を作る。件数が足りなければ空を返す。"""
    entries = prediction_log.read_entries(limit=LIVE_LOOKBACK, kinds=LIVE_KINDS)
    frame = prediction_log.weather_frame(entries)
    return frame if len(frame) >= MIN_LIVE_ROWS else pd.DataFrame(columns=frame.columns)


def _choose_current_source():
    """「現在のデータ」に何を使うかを選ぶ。(表, 出典の名前, 説明) を返す。"""
    live = _live_current_frame()
    if len(live) >= MIN_LIVE_ROWS:
        return live, "live_predictions", (
            f"直近の予測記録 {len(live)} 件を使っています。"
        )

    if os.path.exists(CONFIG.paths.holdout):
        holdout = load_dataset(CONFIG.paths.holdout)
        return holdout, "holdout_data", (
            f"予測の記録がまだ {MIN_LIVE_ROWS} 件に足りないため（現在 {len(live)} 件）、"
            "学習後の未来データで代用しています。使われはじめると自動的に切り替わります。"
        )

    raise MonitorUnavailableError(
        "比較に使えるデータがありません。アプリを使うか、"
        "python fetch_weather.py --holdout でデータを用意してください。"
    )


def monitor_report() -> dict:
    """アプリの現状に合わせて、ドリフトの報告を1つにまとめて返す。"""
    reference = load_dataset(CONFIG.paths.dataset)
    current, source, note = _choose_current_source()

    if len(current) == 0:
        raise MonitorUnavailableError(
            "比較に使えるデータがありません。アプリを使うか、"
            "python fetch_weather.py --holdout でデータを用意してください。"
        )

    service = None
    try:
        service = OutingService(category=load_bundle(CONFIG.paths.category_model))
    except FileNotFoundError:
        pass

    report = monitor.drift_report(reference, current, service)
    report["current_source"] = source
    report["current_note"] = note
    report["min_live_rows"] = MIN_LIVE_ROWS
    return report
