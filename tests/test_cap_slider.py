from app import _apply_runtime_cap


def test_runtime_cap_applies_when_cache_cold():
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    capped, effective = _apply_runtime_cap(symbols, 2, False, False)
    assert capped == symbols[:2]
    assert effective == 2


def test_runtime_cap_bypasses_on_warm_cache():
    symbols = ["AAA", "BBB", "CCC"]
    capped, effective = _apply_runtime_cap(symbols, 1, True, True)
    assert capped == symbols
    assert effective == 0


def test_runtime_cap_honours_override_when_warm():
    symbols = ["AAA", "BBB", "CCC"]
    capped, effective = _apply_runtime_cap(symbols, 1, False, True)
    assert capped == symbols[:1]
    assert effective == 1
