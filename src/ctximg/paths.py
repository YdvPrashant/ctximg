"""Per-user config and data directory resolution.

Every function resolves its answer on each call rather than caching it at import
time, so tests (and users) can redirect the whole application by setting
``CTXIMG_HOME``.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "ctximg"
HOME_ENV = "CTXIMG_HOME"


def _override() -> Path | None:
    value = os.environ.get(HOME_ENV)
    return Path(value) if value else None


def config_dir() -> Path:
    r"""Directory holding config.json (%APPDATA%\ctximg on Windows)."""
    if (root := _override()) is not None:
        return root / "config"
    if os.name == "nt":
        base = os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming"
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / APP_NAME


def data_dir() -> Path:
    r"""Directory holding the index and thumbnails (%LOCALAPPDATA%\ctximg)."""
    if (root := _override()) is not None:
        return root / "data"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
    else:
        base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / APP_NAME


def config_path() -> Path:
    return config_dir() / "config.json"


def db_path() -> Path:
    return data_dir() / "index.db"


def thumbs_dir() -> Path:
    return data_dir() / "thumbs"
