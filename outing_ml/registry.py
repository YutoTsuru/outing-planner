"""モデルの成果物と、その履歴の管理。

学習した .pkl だけを配っても、あとから
「これはいつ・どのデータ・どのコードで作ったモデルなのか」が分かりません。
そこで、モデル本体といっしょに次の情報を必ず保存します。

  ・特徴量の名前と順番（推論時にズレていないか検証するため）
  ・学習データの指紋（sha256）
  ・そのときのコミット（git）
  ・成績とハイパーパラメータ
  ・ライブラリのバージョン

そして model/registry.json に履歴を積み上げ、いつでも前の版に戻せるようにします。
"""

import json
import os
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import joblib

from outing_ml.config import CONFIG

# 成果物の形式のバージョン。読み込み側はこれを見て互換性を判断する
BUNDLE_FORMAT = 2


def git_sha(short: bool = True) -> Optional[str]:
    """いまのコミットのハッシュを返す（git が無い環境では None）。"""
    command = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
    try:
        result = subprocess.run(
            [part for part in command if part != "HEAD"] + ["HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def git_is_dirty() -> Optional[bool]:
    """コミットしていない変更があるかどうか。

    「実験のときのコードが残っていないモデル」を防ぐための印です。
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def environment_info() -> Dict[str, str]:
    """学習したときのライブラリのバージョンを記録する。"""
    import numpy
    import pandas
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "scikit_learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "joblib": joblib.__version__,
    }


@dataclass
class ModelBundle:
    """モデル本体と、それを正しく使うために必要な情報のセット。"""

    estimator: object
    feature_names: List[str]
    model_name: str
    version: str
    task: str
    classes: Optional[List[str]] = None
    target: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    format_version: int = BUNDLE_FORMAT

    def predict_frame(self, frame):
        """特徴量の順番をそろえてから予測する。"""
        return self.estimator.predict(frame[self.feature_names])

    def check_features(self, columns: List[str]) -> None:
        """渡された列が、学習時と同じ名前・同じ順番かを確かめる。"""
        if list(columns) != list(self.feature_names):
            raise ValueError(
                "特徴量が学習時と違います。\n"
                f"  学習時: {self.feature_names}\n"
                f"  渡された: {list(columns)}"
            )


def save_bundle(path: str, bundle: ModelBundle) -> str:
    """成果物を保存する。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str) -> ModelBundle:
    """成果物を読み込む。

    古い形式（モデル本体だけを保存したもの）も読めるようにしておきます。
    本番で「読み込めなくなった」がいちばん困るためです。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"モデルが見つかりません: {path}\n"
            "先に python train_all.py を実行してください。"
        )

    loaded = joblib.load(path)

    if isinstance(loaded, ModelBundle):
        return loaded

    # 古い形式：中身をつつんで返す
    estimator = loaded.get("model") if isinstance(loaded, dict) else loaded
    feature_names = (
        loaded.get("input_columns") if isinstance(loaded, dict) else None
    ) or list(CONFIG.__class__.__dict__.get("FEATURE_COLUMNS", []) or [])

    from outing_ml.config import FEATURE_COLUMNS

    return ModelBundle(
        estimator=estimator,
        feature_names=feature_names or FEATURE_COLUMNS,
        model_name=os.path.basename(path),
        version="legacy",
        task="unknown",
        classes=list(getattr(estimator, "classes_", []) or []) or None,
        format_version=1,
    )


class Registry:
    """学習の履歴を model/registry.json に積み上げる。"""

    def __init__(self, path: str = None):
        self.path = path or CONFIG.paths.registry

    def _read(self) -> List[Dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as file:
            return json.load(file).get("entries", [])

    def record(
        self,
        model_name: str,
        artifact_path: str,
        task: str,
        metrics: Dict,
        data_fingerprint: Dict,
        params: Dict = None,
    ) -> Dict:
        """1回ぶんの学習を記録して、その中身を返す。"""
        entries = self._read()
        previous = [entry for entry in entries if entry["model_name"] == model_name]
        version = f"v{len(previous) + 1}"

        entry = {
            "model_name": model_name,
            "version": version,
            "artifact": artifact_path,
            "task": task,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": git_sha(),
            "git_dirty": git_is_dirty(),
            "data": data_fingerprint,
            "params": params or {},
            "metrics": metrics,
            "environment": environment_info(),
        }

        entries.insert(0, entry)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as file:
            json.dump({"entries": entries}, file, ensure_ascii=False, indent=2)

        return entry

    def latest(self, model_name: str) -> Optional[Dict]:
        """そのモデルの最新の記録を返す。"""
        for entry in self._read():
            if entry["model_name"] == model_name:
                return entry
        return None

    def history(self, model_name: str) -> List[Dict]:
        """そのモデルの記録を新しい順に返す。"""
        return [entry for entry in self._read() if entry["model_name"] == model_name]
