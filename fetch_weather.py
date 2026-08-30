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

import pandas as pd

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


def expected_days(start_date, end_date):
    """期間に含まれる日数。"""
    return (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days + 1


def completed_cities(path, start_date, end_date, tolerance=0.95):
    """すでに取れている都市を返す。

    欠測で数日落ちることがあるので、期待日数の 95% 以上あれば「取れている」とみなします。
    """
    if not os.path.exists(path):
        return set(), None

    frame = pd.read_csv(path)
    if frame.empty:
        return set(), None

    needed = expected_days(start_date, end_date) * tolerance
    counts = frame.groupby("city")["date"].count()
    return set(counts[counts >= needed].index), frame


def save(df, path):
    """検証してから保存する。"""
    report = validate_frame(df)
    if not report.ok:
        print("  検証に失敗しました:")
        for error in report.errors:
            print(f"    - {error}")
        raise SystemExit(1)

    for warning in report.warnings[:3]:
        print(f"  ⚠ {warning}")
    if len(report.warnings) > 3:
        print(f"  ⚠ ほか {len(report.warnings) - 3} 件の警告")

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n保存しました: {path}（{len(df):,} 行 / {df['city'].nunique()} 都市）")


def build(path, start_date, end_date, label, force, resume):
    """1つのデータセットを作る（途中まででも保存する）。"""
    total_cities = len(CONFIG.data.cities)

    if os.path.exists(path) and not force and not resume:
        print(f"すでにあります: {path}（作り直すなら --force、続きから取るなら --resume）")
        return

    done, existing = (set(), None)
    if resume and not force:
        done, existing = completed_cities(path, start_date, end_date)
        if done:
            print(f"\n{label}: {len(done)}/{total_cities} 都市はすでに取得済み。残りを取ります")

    if len(done) >= total_cities:
        print(f"{label}: すべての都市がそろっています（{path}）")
        return

    print(f"\n{label}を取得します（{start_date} 〜 {end_date}）")
    fetched = download_range(start_date, end_date, progress=report_progress,
                             done_cities=done)

    if fetched.empty and existing is None:
        print("  1件も取得できませんでした")
        raise SystemExit(1)

    frames = [frame for frame in (existing, fetched) if frame is not None and not frame.empty]
    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["city", "date"], keep="last")
        .sort_values(["city", "date"])
        .reset_index(drop=True)
    )
    save(combined, path)

    missing = total_cities - combined["city"].nunique()
    if missing:
        print(f"  残り {missing} 都市は未取得です。"
              f"時間をおいて python fetch_weather.py --resume で続きから取れます")


def main():
    parser = argparse.ArgumentParser(description="気象データをダウンロードしてCSVに保存する")
    parser.add_argument("--holdout", action="store_true", help="未来データだけを作る")
    parser.add_argument("--all", action="store_true", help="学習データと未来データの両方を作る")
    parser.add_argument("--force", action="store_true", help="すでにあっても作り直す")
    parser.add_argument("--resume", action="store_true",
                        help="すでに取れている都市は飛ばして、残りだけ取る")
    args = parser.parse_args()

    want_train = args.all or not args.holdout
    want_holdout = args.all or args.holdout

    if want_train:
        build(DATA_PATH, CONFIG.data.start_date, CONFIG.data.end_date,
              "学習データ", args.force, args.resume)

    if want_holdout:
        build(HOLDOUT_PATH, CONFIG.data.holdout_start_date, holdout_end_date(),
              "未来データ（ホールドアウト）", args.force, args.resume)

    print("\n次は python train_all.py でモデルを学習してください。")


if __name__ == "__main__":
    main()
