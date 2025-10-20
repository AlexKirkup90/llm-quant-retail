from app import _format_cap_status


def test_cap_status_bypass_active():
    count, label = _format_cap_status(503, 150, True)
    assert count == 503
    assert label == "none"


def test_cap_status_respects_runtime_cap():
    count, label = _format_cap_status(503, 150, False)
    assert count == 150
    assert label == "150"


def test_cap_status_no_cap_value():
    count, label = _format_cap_status(200, 0, False)
    assert count == 200
    assert label == "none"
