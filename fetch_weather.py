"""
お出かけプランナー：気象データ取得スクリプト

Open-Meteo の「過去の天気アーカイブ」から、日本の主要9都市の実測気象データを
ダウンロードして、1行=1日の CSV にまとめます。

    data/weather_jp.csv          学習・検証に使う（2019〜2024年）
    data/weather_jp_holdout.csv  学習後に一度だけ使う「未来のデータ」（2025年〜）

学習データと未来データを分けているのは、
「学習が終わったあとに一度だけ見る、まったく触っていないデータ」を残しておくためです。
何度も見て調整したデータでは、成績が良く見えて当たり前になってしまいます。

APIキーは不要で、非商用利用は無料です（出典：Open-Meteo / ECMWF ERA5、CC BY 4.0）。

実行方法:
    python fetch_weather.py              # 学習データを作る
    python fetch_weather.py --holdout    # 未来データを作る
    python fetch_weather.py --all        # 両方
    python fetch_weather.py --force      # すでにあっても作り直す
"""

import argparse
import os
from datetime import date, timedelta

from outing_ml.config import CONFIG
from outing_ml.data import validate_frame
from outing_ml.weather_source import download_range

DATA_DIR = CONFIG.paths.data_dir
DATA_PATH = CONFIG.paths.dataset
HOLDOUT_PATH = CONFIG.paths.holdout


def report_progress(index, total, name):
    """取得の進み具合を表示する。"""
    print(f"  [{index}/{total}] {name} を取得中...", flush=True)


def download_all():
    """学習データの期間を取得する（train_model.py から呼ばれる）。"""
    return download_range(CONFIG.data.start_date, CONFIG.data.end_date,
                          progress=report_progress)


def holdout_end_date():
    """未来データの終わりの日。

    アーカイブは数日おくれて更新されるため、直近1週間は取りません。
    """
    return (date.today() - timedelta(days=CONFIG.data.holdout_lag_days)).isoformat()


def save(df, path):
    """検証してから保存する。"""
    report = validate_frame(df)
    if not report.ok:
        print("  検証に失敗しました:")
        for error in report.errors:
            print(f"    - {error}")
        raise SystemExit(1)

    for warning in report.warnings:
        print(f"  ⚠ {warning}")

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n保存しました: {path}（{len(df)} 行）")


def build(path, start_date, end_date, label, force):
    """1つのデータセットを作る。"""
    if os.path.exists(path) and not force:
        print(f"すでにあります: {path}（作り直すときは --force）")
        return

    print(f"\n{label}を取得します（{start_date} 〜 {end_date}）")
    save(download_range(start_date, end_date, progress=report_progress), path)


def main():
    parser = argparse.ArgumentParser(description="気象データをダウンロードしてCSVに保存する")
    parser.add_argument("--holdout", action="store_true", help="未来データだけを作る")
    parser.add_argument("--all", action="store_true", help="学習データと未来データの両方を作る")
    parser.add_argument("--force", action="store_true", help="すでにあっても作り直す")
    args = parser.parse_args()

    want_train = args.all or not args.holdout
    want_holdout = args.all or args.holdout

    if want_train:
        build(DATA_PATH, CONFIG.data.start_date, CONFIG.data.end_date,
              "学習データ", args.force)

    if want_holdout:
        build(HOLDOUT_PATH, CONFIG.data.holdout_start_date, holdout_end_date(),
              "未来データ（ホールドアウト）", args.force)

    print("\n次は python train_all.py でモデルを学習してください。")


if __name__ == "__main__":
    main()
