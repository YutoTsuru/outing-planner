"""ドリフト（データのずれ）監視。

モデルは作った瞬間から古くなります。
気候が変わる、観測のしかたが変わる、使われ方が変わる——
そのたびに、入ってくるデータは学習したときと形が変わっていきます。

このモジュールは、学習に使ったデータ（基準）と、いま入ってきているデータを比べて、
「そろそろ学習し直したほうがよい」という合図を出します。

実行方法:
    python -m outing_ml.monitor
    python -m outing_ml.monitor --current data/weather_jp_holdout.csv
"""

import argparse
import json
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from outing_ml.config import CATEGORIES, CONFIG, FEATURE_COLUMNS
from outing_ml.data import load_dataset

# 0 で割らないための、ごく小さい数
EPSILON = 1e-6


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = None
) -> float:
    """PSI（母集団安定性指標）。分布がどれだけずれたかを1つの数字にする。

    基準データを10個の区間に分け、それぞれの区間に入る割合を比べます。
    目安（金融でよく使われる基準）:
        0.10 未満 … ほぼ変化なし
        0.10〜0.25 … 要観察
        0.25 以上 … 学習し直しを検討
    """
    bins = bins or CONFIG.monitor.psi_bins

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    # 基準データの分位点で区切る（値の偏りに強い）
    edges = np.unique(np.percentile(reference, np.linspace(0, 100, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    reference_share = np.histogram(reference, bins=edges)[0] / len(reference)
    current_share = np.histogram(current, bins=edges)[0] / len(current)

    reference_share = np.clip(reference_share, EPSILON, None)
    current_share = np.clip(current_share, EPSILON, None)

    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def _status(psi: float, ks_statistic: float, spec=None) -> str:
    """PSI とKS統計量から、3段階の状態を決める。

    KS検定の p値 は使いません。件数が多いと、ごくわずかな差でも
    「有意」になってしまい、いつも警告が出る監視になってしまうためです。
    実務では「どれだけずれたか（効果量）」で判断します。
    """
    spec = spec or CONFIG.monitor

    if psi >= spec.psi_alert or ks_statistic >= spec.ks_alert:
        return "ALERT"
    if psi >= spec.psi_watch:
        return "WATCH"
    return "OK"


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> List[Dict]:
    """入力の4項目それぞれについて、ずれを測る。"""
    rows = []

    for column in FEATURE_COLUMNS:
        reference_values = reference[column].to_numpy(dtype=float)
        current_values = current[column].to_numpy(dtype=float)

        psi = population_stability_index(reference_values, current_values)
        ks = stats.ks_2samp(reference_values, current_values)

        rows.append(
            {
                "feature": column,
                "psi": round(psi, 4),
                "ks_statistic": round(float(ks.statistic), 4),
                "ks_p_value": float(ks.pvalue),
                "reference_mean": round(float(reference_values.mean()), 2),
                "current_mean": round(float(current_values.mean()), 2),
                "shift": round(float(current_values.mean() - reference_values.mean()), 2),
                "status": _status(psi, float(ks.statistic)),
            }
        )

    return rows


def prediction_drift(service, reference: pd.DataFrame, current: pd.DataFrame) -> Dict:
    """予測の出方が変わっていないかを見る。

    入力が同じように見えても、予測の内訳（outdoor が減って indoor が増えた等）が
    変わっていれば、下流のプランづくりに影響します。
    """
    reference_pred = service.predict_batch(reference)["category"]
    current_pred = service.predict_batch(current)["category"]

    reference_share = reference_pred.value_counts(normalize=True)
    current_share = current_pred.value_counts(normalize=True)

    shares = {
        name: {
            "reference": round(float(reference_share.get(name, 0.0)), 4),
            "current": round(float(current_share.get(name, 0.0)), 4),
        }
        for name in CATEGORIES
    }

    # 分布どうしの差（対称KLダイバージェンス）を1つの数字にする
    reference_vector = np.array([shares[name]["reference"] for name in CATEGORIES]) + EPSILON
    current_vector = np.array([shares[name]["current"] for name in CATEGORIES]) + EPSILON
    divergence = float(
        np.sum((current_vector - reference_vector) * np.log(current_vector / reference_vector))
    )

    return {
        "share": shares,
        "divergence": round(divergence, 4),
        "status": "ALERT" if divergence >= 0.25 else ("WATCH" if divergence >= 0.10 else "OK"),
    }


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, service=None) -> Dict:
    """入力と予測のずれをまとめたレポートを返す。"""
    features = feature_drift(reference, current)
    statuses = [row["status"] for row in features]

    report = {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "features": features,
    }

    if service is not None:
        report["prediction"] = prediction_drift(service, reference, current)
        statuses.append(report["prediction"]["status"])

    report["overall"] = (
        "ALERT" if "ALERT" in statuses else ("WATCH" if "WATCH" in statuses else "OK")
    )
    report["action"] = {
        "OK": "対応は不要です。",
        "WATCH": "しばらく様子を見てください。連続して WATCH が出たら学習し直しを検討します。",
        "ALERT": "学習し直し（python train_all.py）を検討してください。",
    }[report["overall"]]

    return report


def main():
    parser = argparse.ArgumentParser(description="学習データと新しいデータのずれを調べる")
    parser.add_argument("--reference", default=CONFIG.paths.dataset, help="基準にするデータ")
    parser.add_argument("--current", default=CONFIG.paths.holdout, help="調べたいデータ")
    parser.add_argument("--json", action="store_true", help="結果をJSONで出す")
    args = parser.parse_args()

    reference = load_dataset(args.reference)
    current = load_dataset(args.current)

    service = None
    try:
        from outing_ml.serve import OutingService

        service = OutingService.load()
    except FileNotFoundError:
        pass

    report = drift_report(reference, current, service)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(f"基準: {args.reference}（{report['reference_rows']} 行）")
    print(f"比較: {args.current}（{report['current_rows']} 行）\n")
    print(pd.DataFrame(report["features"]).to_string(index=False))

    if "prediction" in report:
        print("\n予測の内訳:")
        for name, share in report["prediction"]["share"].items():
            print(f"  {name:<8} 学習時 {share['reference'] * 100:5.1f}% "
                  f"→ いま {share['current'] * 100:5.1f}%")
        print(f"  ずれの大きさ: {report['prediction']['divergence']}"
              f"（{report['prediction']['status']}）")

    print(f"\n総合判定: {report['overall']}")
    print(report["action"])


if __name__ == "__main__":
    main()
