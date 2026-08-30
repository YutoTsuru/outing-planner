"""レート制限（rate_limit.py）のテスト。"""

import pytest

import rate_limit


@pytest.fixture(autouse=True)
def clean_state():
    rate_limit.reset()
    yield
    rate_limit.reset()


def test_上限までは許可される():
    for _ in range(5):
        result = rate_limit.check("client-a", limit=5, now=1000.0)
        assert result.allowed

    assert result.remaining == 0


def test_上限を超えると拒否される():
    for _ in range(5):
        rate_limit.check("client-a", limit=5, now=1000.0)

    result = rate_limit.check("client-a", limit=5, now=1000.0)

    assert not result.allowed
    assert result.remaining == 0
    assert result.reset_seconds > 0


def test_識別子が違えば別々に数える():
    for _ in range(5):
        rate_limit.check("client-a", limit=5, now=1000.0)

    result = rate_limit.check("client-b", limit=5, now=1000.0)
    assert result.allowed


def test_窓が変わるとリセットされる():
    for _ in range(5):
        rate_limit.check("client-a", limit=5, now=1000.0)   # 1000秒台の窓

    result = rate_limit.check("client-a", limit=5, now=1065.0)   # 60秒後の次の窓
    assert result.allowed


def test_拒否された分は数を消費しない():
    for _ in range(6):
        rate_limit.check("client-a", limit=5, now=1000.0)

    # 6回目以降ずっと拒否されていても、状態が壊れて許可に転じたりしない
    result = rate_limit.check("client-a", limit=5, now=1000.0)
    assert not result.allowed


def test_古い窓のカウントを掃除できる():
    rate_limit.check("client-a", limit=5, now=1000.0)
    rate_limit.check("client-a", limit=5, now=2000.0)

    removed = rate_limit.prune_old_windows(now=2000.0)

    assert removed == 1   # 1000秒台の窓だけが古いので消える
    # 現在の窓のカウントは残っている
    result = rate_limit.check("client-a", limit=1, now=2000.0)
    assert not result.allowed   # すでに1回使っているので、上限1なら拒否される
