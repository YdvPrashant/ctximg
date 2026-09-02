"""The desktop app: one server on a fixed port, a tray icon, and a browser tab.

Launched as `pythonw -m ctximg.desktop` so no console window appears. Clicking
the icon while it is already running must not start a second server, so the
port itself is the lock: if 8077 is taken by a live ctximg, we just open a tab.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import traceback
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from . import paths

HOST = "127.0.0.1"
PORT = 8077
MARKER = "ctximg"


def log_path() -> Path:
    return paths.data_dir() / "app.log"


def _ensure_streams() -> Path:
    """Give the process real stdout/stderr before anything needs them.

    pythonw hands a process None for stdout, stderr and stdin. Libraries
    reasonably assume otherwise - uvicorn's log formatter calls
    sys.stdout.isatty() while starting - so without this the app dies before
    the server exists, and dies invisibly because there is nowhere to report
    it. Pointing them at a log file fixes the crash and makes the next one
    findable.
    """
    log = log_path()
    if sys.stdout is not None and sys.stderr is not None and sys.stdin is not None:
        return log

    log.parent.mkdir(parents=True, exist_ok=True)
    stream = open(log, "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream
    if sys.stdin is None:
        sys.stdin = open(os.devnull, "r")
    return log


def base_url(port: int = PORT) -> str:
    return f"http://{HOST}:{port}"


def _already_running(port: int = PORT, timeout: float = 1.5) -> bool:
    """True if a ctximg app is answering on the port.

    Checked by asking rather than by trying to bind, so another program
    squatting the port is not mistaken for our own app.
    """
    try:
        with urllib.request.urlopen(f"{base_url(port)}/api/ping", timeout=timeout) as r:
            return r.read(64).decode("utf-8", "ignore").strip().startswith(MARKER)
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _port_free(port: int = PORT) -> bool:
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            probe.bind((HOST, port))
            return True
        except OSError:
            return False


def app_url(folder_key: str | None = None) -> str:
    return base_url() + (f"/?folder={folder_key}" if folder_key else "/")


def ensure_running(folder_key: str | None = None, open_browser: bool = True) -> str | None:
    """Make sure the app is up, then open it. Returns the URL, or None.

    Used both by the desktop icon and by `ctximg web`, so the terminal and the
    icon can never end up running two competing servers.
    """
    url = app_url(folder_key)
    if _already_running():
        if open_browser:
            webbrowser.open(url)
        return url

    if not _port_free():
        return None

    started = threading.Event()
    thread = threading.Thread(
        target=_serve_forever, args=(started,), daemon=True, name="ctximg-app"
    )
    thread.start()
    if not started.wait(timeout=30):
        return None
    if open_browser:
        webbrowser.open(url)
    return url


def _build_server():
    import uvicorn

    from .app import App
    from .web.server import create_app

    _ensure_streams()
    config = uvicorn.Config(
        create_app(App()),
        host=HOST,
        port=PORT,
        log_level="warning",
        # Skip uvicorn's own logging setup: its formatter inspects sys.stdout,
        # which is not something a windowless process is guaranteed to have.
        log_config=None,
    )
    server = uvicorn.Server(config)
    # Only the main thread may install signal handlers, and this often is not it.
    server.install_signal_handlers = lambda: None
    return server


def _serve_forever(started: threading.Event | None = None) -> None:
    server = _build_server()
    if started is not None:
        def watch():
            while not server.started:
                if not threading.main_thread().is_alive():
                    return
                threading.Event().wait(0.1)
            started.set()

        threading.Thread(target=watch, daemon=True).start()
    server.run()


def _run_tray(server) -> None:
    """Tray icon with Open and Quit. Falls back to running headless."""
    try:
        import pystray
        from PIL import Image, ImageDraw
    except Exception:
        return None

    image = Image.new("RGB", (64, 64), (19, 19, 20))
    draw = ImageDraw.Draw(image)
    draw.ellipse((16, 16, 48, 48), outline=(237, 237, 239), width=5)
    draw.line((44, 44, 56, 56), fill=(237, 237, 239), width=6)

    def on_open(icon, item):
        webbrowser.open(app_url())

    def on_quit(icon, item):
        server.should_exit = True
        icon.stop()

    return pystray.Icon(
        "ctximg",
        image,
        "ctximg - photo search",
        menu=pystray.Menu(
            pystray.MenuItem("Open ctximg", on_open, default=True),
            pystray.MenuItem("Quit", on_quit),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the desktop icon.

    Everything is wrapped so a windowless launch can never fail in silence -
    the desktop icon simply doing nothing is the worst possible failure.
    """
    log = _ensure_streams()
    try:
        return _main(argv)
    except BaseException:
        try:
            with open(log, "a", encoding="utf-8") as handle:
                handle.write("\n=== ctximg failed to start ===\n")
                handle.write(traceback.format_exc())
        except OSError:
            pass
        raise


def _main(argv: list[str] | None = None) -> int:
    if _already_running():
        webbrowser.open(app_url())
        return 0
    if not _port_free():
        print(f"Port {PORT} is in use by something else.", file=sys.stderr)
        return 1

    server = _build_server()
    threading.Thread(target=server.run, daemon=True, name="ctximg-app").start()

    for _ in range(300):
        if server.started:
            break
        threading.Event().wait(0.1)

    webbrowser.open(app_url())

    icon = _run_tray(server)
    if icon is None:
        # No tray available: stay alive until the server stops.
        while not server.should_exit:
            threading.Event().wait(0.5)
        return 0
    icon.run()  # blocks until Quit
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
