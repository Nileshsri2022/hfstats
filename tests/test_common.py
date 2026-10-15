"""Tests for scripts/common.py helpers."""
import common


def test_gh_annotation_is_noop_outside_actions(capsys):
    common.gh_annotation("error", "boom", title="T")
    assert capsys.readouterr().out == ""


def test_gh_annotation_escapes_message_and_title(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    common.gh_annotation("warning", "50% failed\nsecond line", title="a: b, c")
    assert capsys.readouterr().out == "::warning title=a%3A b%2C c::50%25 failed%0Asecond line\n"


def test_gh_annotation_without_title(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    common.gh_annotation("notice", "hello")
    assert capsys.readouterr().out == "::notice::hello\n"
