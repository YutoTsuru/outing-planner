"""
お出かけプランナー：4つのモデルをまとめて学習するスクリプト

このプロジェクトには4つのモデルがあります。
それぞれ別々に実行してもよいのですが、順番に気をつける必要があるため
（カテゴリ予測モデルを、あとの2つが参照します）、
このスクリプトで正しい順に一気に作れるようにしています。

    1. カテゴリ予測      train_model.py         → outing_model.pkl
    2. 翌日の天気予測    train_forecast.py      → forecast_model.pkl
    3. おでかけ日和度    train_comfort.py       → comfort_model.pkl
    4. 天気タイプ分け    train_weather_types.py → weather_type_model.pkl

学習データ（data/weather_jp.csv）が無ければ、最初に自動でダウンロードします。

実行方法:
    python train_all.py
"""

import time

import train_comfort
import train_forecast
import train_model
import train_weather_types

# 実行する順番（あとの3つは、1番のカテゴリ予測モデルを参照する）
STEPS = [
    ("カテゴリ予測モデル", train_model),
    ("翌日の天気予測モデル", train_forecast),
    ("おでかけ日和度モデル", train_comfort),
    ("天気タイプ分けモデル", train_weather_types),
]


def main():
    started = time.time()

    for number, (name, module) in enumerate(STEPS, start=1):
        print("\n" + "=" * 60)
        print(f"  [{number}/{len(STEPS)}] {name}")
        print("=" * 60 + "\n")
        module.main()

    print("\n" + "=" * 60)
    print(f"  すべて完了しました（{time.time() - started:.0f} 秒）")
    print("=" * 60)
    print("\nモデルの説明は doc/README.md にまとめてあります。")
    print("次は python webapp.py でアプリを起動してください。")


if __name__ == "__main__":
    main()
