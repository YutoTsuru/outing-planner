"""IPアドレス単位のレート制限（固定窓カウンタ）。

flask-limiter のような外部ライブラリは使わず、プロセス内のメモリだけで
数えます。教材アプリの規模ではこれで十分で、依存を増やさずに済みます。

固定窓カウンタは「1分ちょうど」で数えをリセットする方式です。厳密な
トークンバケットより単純ですが、窓の境目でまとめて叩かれると瞬間的に
2倍近く通ることがあります（例：0:59に59回、1:01に59回）。教材用の
簡易な安全弁としてはこの単純さのほうが分かりやすいと判断しています。

このカウンタはプロセス内のメモリだけにあります。gunicorn を複数ワーカーで
動かすと、ワーカーごとに別々に数えます（1ワーカーあたりの上限になる）。
"""

import time
from collections import defaultdict
from dataclasses import dataclass

# 既定の上限（1分あたりの回数）。エンドポイントごとに変えられる。
DEFAULT_LIMIT_PER_MINUTE = 60

WINDOW_SECONDS = 60

# {(識別子, 窓の開始時刻): 回数}
_counts: dict[tuple[str, int], int] = defaultdict(int)


@dataclass
class RateLimitResult:
    """判定の結果。"""

    allowed: bool
    limit: int
    remaining: int
    reset_seconds: int   # 次の窓まであと何秒か


def _current_window(now: float) -> int:
    """いまが、何番目の「1分間の窓」に入っているか。"""
    return int(now // WINDOW_SECONDS)


def check(key: str, limit: int = DEFAULT_LIMIT_PER_MINUTE, now: float = None) -> RateLimitResult:
    """key（多くはIPアドレス＋エンドポイント名）の残り回数を確認し、1回分消費する。

    呼ぶたびに1回消費します。制限に達している判定のときも消費はしません
    （すでに超えている相手にこれ以上カウントを積む意味が無いため）。
    """
    now = time.time() if now is None else now
    window = _current_window(now)
    window_key = (key, window)

    used = _counts[window_key]
    if used >= limit:
        reset_seconds = int((window + 1) * WINDOW_SECONDS - now) + 1
        return RateLimitResult(allowed=False, limit=limit, remaining=0,
                               reset_seconds=reset_seconds)

    _counts[window_key] = used + 1
    reset_seconds = int((window + 1) * WINDOW_SECONDS - now) + 1
    return RateLimitResult(allowed=True, limit=limit, remaining=limit - used - 1,
                           reset_seconds=reset_seconds)


def reset() -> None:
    """カウンタを空にする（テスト用）。"""
    _counts.clear()


def prune_old_windows(now: float = None) -> int:
    """過去の窓のカウントを掃除する。メモリが際限なく増えるのを防ぐ。

    リクエストのたびに毎回全件を掃除すると重いので、外部から時々
    （例えば一定間隔で）呼び出す想定です。戻り値は消した件数。
    """
    now = time.time() if now is None else now
    current = _current_window(now)
    stale = [key for key in _counts if key[1] < current]
    for key in stale:
        del _counts[key]
    return len(stale)
