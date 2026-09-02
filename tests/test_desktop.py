"""Launching without a console.

pythonw hands a process None for stdout, stderr and stdin. Libraries assume
otherwise - uvicorn's log formatter calls sys.stdout.isatty() while starting -
so a windowless launch used to die before the server existed, and die silently
because there was nowhere to report it. The desktop icon simply doing nothing
is the worst possible failure, so it is worth a test.
"""

from __future__ import annotations

import sys


from ctximg import desktop


def go_headless(monkeypatch):
    """Strip the process of its streams, the way pythonw starts one.

    Called inside the test body, never from a fixture: pytest re-installs
    sys.stdout when it resumes capture for the call phase, which would undo a
    fixture's patch and leave the test passing without ever testing anything.
    """
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(sys, "stdin", None)


def test_streams_are_replaced_when_missing(monkeypatch):
    go_headless(monkeypatch)
    log = desktop._ensure_streams()

    assert sys.stdout is not None
    assert sys.stderr is not None
    assert sys.stdin is not None
    assert log.name == "app.log"
    assert log.exists()

    # Asserted on where the stream points rather than by reading the file back:
    # under pytest's fd-level capture the write is intercepted, which would test
    # the harness rather than this code.
    assert sys.stdout.name == str(log)
    assert sys.stderr.name == str(log)
    assert sys.stdout.writable()


def test_real_streams_are_left_alone():
    before = (sys.stdout, sys.stderr, sys.stdin)
    desktop._ensure_streams()
    assert (sys.stdout, sys.stderr, sys.stdin) == before


def test_server_builds_with_no_streams(monkeypatch):
    """The actual regression: this raised ValueError under pythonw."""
    go_headless(monkeypatch)
    assert sys.stdout is None, "the test must really be headless to mean anything"
    server = desktop._build_server()
    assert server is not None
    assert server.config.port == desktop.PORT
    assert server.config.host == "127.0.0.1"


def test_server_does_not_hand_logging_a_stdout_it_may_not_have():
    server = desktop._build_server()
    assert server.config.log_config is None


def test_signal_handlers_are_not_installed_off_the_main_thread():
    """Only the main thread may install them, and the server runs beside a UI."""
    server = desktop._build_server()
    assert server.install_signal_handlers() is None


def test_app_url_carries_a_requested_folder():
    assert desktop.app_url() == "http://127.0.0.1:8077/"
    assert desktop.app_url("abc123").endswith("/?folder=abc123")


def test_nothing_running_is_reported_honestly(monkeypatch):
    import urllib.error

    def refuse(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(desktop.urllib.request, "urlopen", refuse)
    assert desktop._already_running() is False


def test_a_stranger_on_the_port_is_not_mistaken_for_us(monkeypatch):
    """Only our own app answers the marker, so we never adopt someone's server."""

    class Response:
        def read(self, n=None):
            return b"<html>some other app</html>"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(desktop.urllib.request, "urlopen", lambda *a, **k: Response())
    assert desktop._already_running() is False


def test_our_own_app_is_recognised(monkeypatch):
    class Response:
        def read(self, n=None):
            return b"ctximg"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(desktop.urllib.request, "urlopen", lambda *a, **k: Response())
    assert desktop._already_running() is True
