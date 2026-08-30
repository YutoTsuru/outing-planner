"""お出かけプランの共有リンク。

/plan で作ったプランに短いIDを振り、そのIDだけで後からもう一度開けるようにします。
永続的なデータベースは無いプロジェクトなので、prediction_log.py と同じ
JSON Lines形式で1行ずつ追記します（読み込みも同じ考え方）。

有効期限やアクセス制限はありません。教材用の簡易な仕組みで、
IDを知っている人なら誰でも開けます（個人情報を含むプランは作らない前提）。
"""

import json
import os
import uuid
from datetime import UTC, datetime

from outing_ml.config import CONFIG

# 1行1件の JSON Lines。追記しかしない
STORE_PATH = os.path.join(CONFIG.paths.report_dir, "shared_plans.jsonl")

# 読み込む行数の上限（探すIDが後ろのほうにあっても、際限なく読み続けないため）
MAX_SCAN_LINES = 20000


class PlanNotFoundError(KeyError):
    """指定されたIDのプランが見つからないときに投げる例外。"""


def save(category: str, area: str, plan_markdown: str) -> str:
    """プランを1件保存し、そのIDを返す。"""
    plan_id = uuid.uuid4().hex[:12]   # 短く、それでいて衝突しない程度の長さ

    entry = {
        "id": plan_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "category": category,
        "area": area,
        "plan_markdown": plan_markdown,
    }

    os.makedirs(os.path.dirname(STORE_PATH) or ".", exist_ok=True)
    with open(STORE_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return plan_id


def find(plan_id: str) -> dict:
    """IDからプランを1件探す。見つからなければ PlanNotFoundError。

    後ろに追記していく形式なので、新しいものほど後ろにあります。
    多くの場合は最近作ったプランを開くはずなので、**後ろから**探します。
    """
    if not os.path.exists(STORE_PATH):
        raise PlanNotFoundError(plan_id)

    with open(STORE_PATH, encoding="utf-8") as file:
        lines = file.readlines()[-MAX_SCAN_LINES:]

    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("id") == plan_id:
            return entry

    raise PlanNotFoundError(plan_id)
