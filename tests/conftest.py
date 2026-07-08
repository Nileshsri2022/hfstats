"""Shared test fixtures and helpers for the HFStats test suite."""
import io
import os
import sys
import urllib.error

import pytest

# Make the modules under scripts/ importable as top-level modules.
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


class FakeResponse:
    """Minimal stand-in for the object returned by urllib.request.urlopen.

    Supports use as a context manager, .read(), .headers.get(), and line
    iteration (used by the streaming benchmark).
    """

    def __init__(self, body="", headers=None, lines=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self._body = body
        self.headers = headers or {}
        self._lines = lines

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        if self._lines is not None:
            for line in self._lines:
                yield line if isinstance(line, bytes) else line.encode("utf-8")
        else:
            for line in self._body.splitlines(keepends=True):
                yield line


def make_http_error(code, body="", headers=None):
    """Construct a urllib.error.HTTPError with a readable body and headers."""
    fp = io.BytesIO(body.encode("utf-8") if isinstance(body, str) else body)
    return urllib.error.HTTPError(
        "https://example.test", code, "err", headers or {}, fp
    )


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point db_utils at a fresh temporary SQLite database."""
    import db_utils

    db_file = tmp_path / "test.db"
    monkeypatch.setattr(db_utils, "DB_PATH", str(db_file))
    db_utils.init_db()
    return db_utils
