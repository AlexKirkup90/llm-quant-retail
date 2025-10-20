from __future__ import annotations

import sys
from types import ModuleType


class _Tab:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class DummyStreamlit(ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.used_keys: set[str] = set()

    def _register(self, key: str | None) -> None:
        if key is None:
            raise AssertionError("All widgets must define a key")
        if key in self.used_keys:
            raise AssertionError(f"Duplicate key detected: {key}")
        self.used_keys.add(key)

    # Widget methods -----------------------------------------------------
    def date_input(self, *_, key=None, **__):
        self._register(key)
        return None

    def selectbox(self, *_, key=None, **__):
        self._register(key)
        return None

    def checkbox(self, *_, key=None, **__):
        self._register(key)
        return False

    def button(self, *_, key=None, **__):
        self._register(key)
        return False

    def slider(self, *_, key=None, **__):
        self._register(key)
        return 0

    def number_input(self, *_, key=None, **__):
        self._register(key)
        return 0.0

    def download_button(self, *_, key=None, **__):
        self._register(key)
        return None

    def tabs(self, names):
        return [_Tab(name) for name in names]

    def expander(self, *_, **__):
        return _Tab("expander")

    # No-op UI helpers ---------------------------------------------------
    def title(self, *_, **__):
        return None

    def subheader(self, *_, **__):
        return None

    def markdown(self, *_, **__):
        return None

    def caption(self, *_, **__):
        return None

    def json(self, *_, **__):
        return None

    def success(self, *_, **__):
        return None

    def error(self, *_, **__):
        return None

    def stop(self, *_, **__):
        return None

    def write(self, *_, **__):
        return None

    def info(self, *_, **__):
        return None

    def dataframe(self, *_, **__):
        return None

    def line_chart(self, *_, **__):
        return None

    def metric(self, *_, **__):
        return None


def test_streamlit_widgets_have_unique_keys(monkeypatch):
    dummy = DummyStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", dummy)

    # Importing the app should not trigger Streamlit usage.
    import app  # noqa: F401

    # Executing main() ensures widgets register their keys.
    app.main()

    assert dummy.used_keys, "Expected at least one widget key to be registered"
