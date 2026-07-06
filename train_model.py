"""
お出かけプランナー：モデル学習スクリプト

気象データ（気温・降水確率・風速・湿度）から、
おすすめのお出かけカテゴリ（outdoor / indoor / relax）を予測する
機械学習モデルを学習して保存します。

授業用のため、外部データセットは使わず、
サンプルデータをこのスクリプト内で作成します。

実行方法:
    python train_model.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 乱数の種（毎回同じサンプルデータを作るために固定する）
RANDOM_SEED = 42

# モデルの保存先
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "outing_model.pkl")

# 特徴量の列名（app.py 側と同じ順番で使うこと）
FEATURE_COLUMNS = ["temperature", "rain_probability", "wind_speed", "humidity"]


def assign_label(temperature, rain_probability, wind_speed, humidity):
    """1件分の気象データに対して、お出かけカテゴリを決めるルール。

    このルールで付けたラベルを「正解データ」としてモデルに学習させます。
    """
    # 雨が降りそうな日は屋内へ
    if rain_probability >= 50:
        return "indoor"

    # 風が強い日も屋内が安心
    if wind_speed >= 10:
        return "indoor"

    # 湿度が高い日や寒い日は、ゆっくり過ごすのがおすすめ
    if humidity >= 80 or temperature < 10:
        return "relax"

    # 過ごしやすい気温なら屋外へ
    if temperature <= 30:
        return "outdoor"

    # 30℃を超える暑い日はリラックス
    return "relax"


def create_sample_data(n_samples=600):
    """学習用のサンプルデータ（DataFrame）を作成する。"""
    rng = np.random.default_rng(RANDOM_SEED)

    # 降水確率は「低い日が多い」ように、2つの乱数の小さい方を採用する
    rain = np.minimum(rng.integers(0, 101, n_samples), rng.integers(0, 101, n_samples))

    df = pd.DataFrame(
        {
            "temperature": np.round(rng.uniform(-5, 38, n_samples), 1),      # 気温（℃）
            "rain_probability": rain,                                        # 降水確率（%）
            "wind_speed": np.round(rng.uniform(0, 12, n_samples), 1),        # 風速（m/s）
            "humidity": rng.integers(20, 101, n_samples),                    # 湿度（%）
        }
    )

    # 1行ずつルールを適用して、ラベル（正解）の列を追加する
    df["label"] = [
        assign_label(row.temperature, row.rain_probability, row.wind_speed, row.humidity)
        for row in df.itertuples()
    ]

    return df


def train_model(df):
    """DataFrame からランダムフォレストのモデルを学習して返す。"""
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    # 学習用データと評価用データに分ける（8:2）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    # 評価用データで精度を確認する
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"テストデータでの正解率: {accuracy:.2f}")

    return model


def save_model(model):
    """学習済みモデルをファイルに保存する。"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"モデルを保存しました: {MODEL_PATH}")


def main():
    print("1. サンプルデータを作成中...")
    df = create_sample_data()
    print(f"   データ件数: {len(df)}")
    print("   カテゴリごとの件数:")
    print(df["label"].value_counts().to_string())

    print("\n2. モデルを学習中...")
    model = train_model(df)

    print("\n3. モデルを保存中...")
    save_model(model)

    print("\n完了！ 次は python app.py でアプリを起動してください。")


if __name__ == "__main__":
    main()
