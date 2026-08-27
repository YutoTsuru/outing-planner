"""評価。

点数を1つ出して終わりにはしません。実務で必要なのは次の4つです。

  1. 信頼区間  … その数字は、たまたま良かっただけではないか
  2. 有意差    … ベースラインとの差は、誤差の範囲ではないか
  3. 内訳      … 都市別・季節別で、ひどく苦手なところは無いか
  4. 確率の質  … 「80%」と言ったとき、本当に8割当たっているか（較正）
"""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from outing_ml.config import CONFIG

# ---------------------------------------------------------------
# 信頼区間
# ---------------------------------------------------------------

def bootstrap_interval(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: Sequence,
    y_pred: Sequence,
    n_samples: int = None,
    seed: int = None,
    alpha: float = 0.05,
) -> dict[str, float]:
    """ブートストラップ法で、指標の95%信頼区間を出す。

    評価用データから重複ありで選び直したものを何度も作り、
    そのたびに指標を計算して、値のちらばりを見ます。
    「正解率 0.817」だけでは、それがどれくらい確かな数字か分かりません。
    """
    n_samples = n_samples or CONFIG.train.bootstrap_samples
    seed = CONFIG.train.random_seed if seed is None else seed

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)

    values = []
    for _ in range(n_samples):
        index = rng.integers(0, len(y_true), len(y_true))
        # 選び直した結果、クラスが1つしか無いと計算できない指標があるため飛ばす
        if len(np.unique(y_true[index])) < 2:
            continue
        values.append(metric(y_true[index], y_pred[index]))

    values = np.asarray(values)
    return {
        "value": float(metric(y_true, y_pred)),
        "low": float(np.percentile(values, 100 * alpha / 2)),
        "high": float(np.percentile(values, 100 * (1 - alpha / 2))),
    }


# ---------------------------------------------------------------
# 分類の評価
# ---------------------------------------------------------------

def classification_metrics(
    y_true: Sequence, y_pred: Sequence, proba: np.ndarray = None, classes: list[str] = None
) -> dict[str, object]:
    """正解率・マクロF1（信頼区間つき）・対数損失・混同行列をまとめて返す。"""
    result = {
        "accuracy": bootstrap_interval(accuracy_score, y_true, y_pred),
        "macro_f1": bootstrap_interval(
            lambda a, b: f1_score(a, b, average="macro"), y_true, y_pred
        ),
    }

    if classes is not None:
        result["confusion_matrix"] = confusion_matrix(
            y_true, y_pred, labels=classes
        ).tolist()

    if proba is not None and classes is not None:
        result["log_loss"] = float(log_loss(y_true, proba, labels=classes))

    return result


def expected_calibration_error(
    y_true: Sequence, proba: np.ndarray, classes: list[str], bins: int = None
) -> dict[str, object]:
    """確率がどれくらい正直かを測る（ECE）。

    「85%の自信がある」と言った予測を集めたとき、
    本当に85%くらい当たっていれば、その確率は信頼できます。
    ずれの平均が ECE で、0に近いほど良い値です。
    """
    bins = bins or CONFIG.train.calibration_bins

    classes = list(classes)
    confidence = proba.max(axis=1)
    predicted = np.asarray([classes[index] for index in proba.argmax(axis=1)])
    correct = (predicted == np.asarray(y_true)).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    table = []
    error = 0.0

    for start, end in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > start) & (confidence <= end)
        count = int(in_bin.sum())
        if count == 0:
            continue

        bin_confidence = float(confidence[in_bin].mean())
        bin_accuracy = float(correct[in_bin].mean())
        error += count / len(y_true) * abs(bin_accuracy - bin_confidence)
        table.append(
            {
                "range": f"{start:.1f}-{end:.1f}",
                "count": count,
                "confidence": round(bin_confidence, 4),
                "accuracy": round(bin_accuracy, 4),
                "gap": round(bin_accuracy - bin_confidence, 4),
            }
        )

    return {"ece": float(error), "bins": table}


def mcnemar_test(y_true: Sequence, pred_a: Sequence, pred_b: Sequence) -> dict[str, object]:
    """2つのモデルの差が、誤差の範囲かどうかを調べる（マクネマー検定）。

    「Aのほうが正解率が1%高い」だけでは、たまたまかもしれません。
    Aだけが当てた件数とBだけが当てた件数を比べて、偏りが本物かを見ます。
    p値が 0.05 より小さければ、差は本物と判断します。
    """
    y_true = np.asarray(y_true)
    a_correct = np.asarray(pred_a) == y_true
    b_correct = np.asarray(pred_b) == y_true

    only_a = int((a_correct & ~b_correct).sum())
    only_b = int((~a_correct & b_correct).sum())

    if only_a + only_b == 0:
        return {"only_a": 0, "only_b": 0, "p_value": 1.0, "significant": False}

    p_value = float(stats.binomtest(only_a, only_a + only_b, 0.5).pvalue)
    return {
        "only_a": only_a,
        "only_b": only_b,
        "p_value": p_value,
        "significant": bool(p_value < 0.05),
    }


# ---------------------------------------------------------------
# 内訳（スライス）
# ---------------------------------------------------------------

def slice_report(
    frame: pd.DataFrame, y_true: Sequence, y_pred: Sequence, by: str, min_count: int = 30
) -> list[dict[str, object]]:
    """都市別・季節別など、グループごとの成績を出す。

    全体の平均が良くても、特定の都市や季節だけ極端に悪いことがあります。
    平均だけを見ていると、それに気づけません。
    """
    work = pd.DataFrame({"group": frame[by].to_numpy(), "true": np.asarray(y_true),
                         "pred": np.asarray(y_pred)})

    rows = []
    for group, part in work.groupby("group"):
        if len(part) < min_count:
            continue
        rows.append(
            {
                "group": str(group),
                "count": int(len(part)),
                "accuracy": float(accuracy_score(part["true"], part["pred"])),
                "macro_f1": float(f1_score(part["true"], part["pred"], average="macro")),
            }
        )

    return sorted(rows, key=lambda row: row["accuracy"])


def season_of(dates: Sequence) -> np.ndarray:
    """日付を、春・夏・秋・冬の4つに分ける。"""
    month = pd.to_datetime(pd.Series(list(dates))).dt.month
    names = {12: "冬", 1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春",
             6: "夏", 7: "夏", 8: "夏", 9: "秋", 10: "秋", 11: "秋"}
    return month.map(names).to_numpy()


# ---------------------------------------------------------------
# 回帰の評価
# ---------------------------------------------------------------

def regression_metrics(y_true: Sequence, y_pred: Sequence, with_interval: bool = True) -> dict:
    """MAE・RMSE・R2 を返す（MAE には信頼区間をつける）。"""
    result = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }

    if with_interval:
        interval = bootstrap_interval(mean_absolute_error, y_true, y_pred, n_samples=300)
        result["mae_low"] = interval["low"]
        result["mae_high"] = interval["high"]

    return result


def pinball_loss(y_true: Sequence, y_pred: Sequence, quantile: float) -> float:
    """分位点予測の誤差（ピンボール損失）。予測区間の良し悪しを測る。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    difference = y_true - y_pred
    return float(np.mean(np.maximum(quantile * difference, (quantile - 1) * difference)))


def interval_coverage(y_true: Sequence, lower: Sequence, upper: Sequence) -> float:
    """予測区間の中に、実際の値がどれくらい入ったか。

    「80%の区間」なら、だいたい80%入っているのが正しい姿です。
    入りすぎ（区間が広すぎ）も、入らなすぎ（自信過剰）も問題になります。
    """
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= np.asarray(lower, dtype=float)) & (
        y_true <= np.asarray(upper, dtype=float)
    )
    return float(inside.mean())
