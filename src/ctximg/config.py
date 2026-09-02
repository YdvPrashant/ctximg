"""Persistent configuration.

The gallery folders are configured once and stay put until changed; every
command (index, search, serve) reads them from here rather than taking a path
argument.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from . import paths

IMAGE_EXTENSIONS = [
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif",
    ".tif", ".tiff", ".heic", ".heif",
]

IGNORE_DIRS = [
    ".git", "node_modules", "$RECYCLE.BIN", "System Volume Information",
    ".thumbnails", "__pycache__", ".cache",
]


class ConfigError(ValueError):
    """A configuration change the user asked for cannot be applied."""


@dataclass
class Config:
    # Workspace keys ticked in the app. Empty means "all of them", so a fresh
    # install searches everything rather than nothing.
    selected: list[str] = field(default_factory=list)
    tier: str = "fast"                # fast | balanced | best
    precision: str = "auto"           # auto | fp16 | fp32
    # Measured images/second per "model|device", so estimates get truthful
    # once this machine has actually done the work.
    rates: dict = field(default_factory=dict)
    model: str | None = None          # None -> take the tier's model
    device: str = "auto"              # auto | cuda | cpu
    thumb_size: int = 256
    batch_size: int | None = None     # None -> auto (32 CPU / 64 GPU)
    extensions: list[str] = field(default_factory=lambda: list(IMAGE_EXTENSIONS))
    ignore_dirs: list[str] = field(default_factory=lambda: list(IGNORE_DIRS))

    # --- generic key access ------------------------------------------------

    def set_value(self, key: str, raw: str) -> object:
        """Set a scalar/list option from its string form. Returns the new value."""
        if key in ("selected", "rates"):
            raise ConfigError(f"{key!r} is managed by the app, not set by hand")
        known = {f.name for f in fields(self)}
        if key not in known:
            raise ConfigError(f"Unknown option '{key}'. Known: {', '.join(sorted(known))}")

        value = _coerce(key, raw)
        if key == "device" and value not in ("auto", "cuda", "cpu"):
            raise ConfigError("device must be one of: auto, cuda, cpu")
        if key == "tier" and value not in ("fast", "balanced", "best"):
            raise ConfigError("tier must be one of: fast, balanced, best")
        if key == "precision" and value not in ("auto", "fp16", "fp32"):
            raise ConfigError("precision must be one of: auto, fp16, fp32")
        if key == "thumb_size" and (not isinstance(value, int) or value < 32):
            raise ConfigError("thumb_size must be an integer >= 32")
        if key == "batch_size" and value is not None and value < 1:
            raise ConfigError("batch_size must be >= 1, or 'auto'")
        setattr(self, key, value)
        return value

    # --- serialisation -----------------------------------------------------

    def remember_rate(self, model_id: str, device: str, rate: float) -> None:
        """Record how fast this machine really is, for honest estimates."""
        if rate > 0:
            self.rates[f"{model_id}|{device}"] = round(rate, 2)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


_NULLISH = {"", "auto", "none", "null", "default"}


def _coerce(key: str, raw: str) -> object:
    if key in ("model", "batch_size") and raw.strip().lower() in _NULLISH:
        return None
    if key in ("extensions", "ignore_dirs", "selected"):
        items = [p.strip() for p in raw.split(",") if p.strip()]
        if key == "extensions":
            items = [i if i.startswith(".") else "." + i for i in items]
            items = [i.lower() for i in items]
        return items
    if key in ("thumb_size", "batch_size"):
        try:
            return int(raw)
        except ValueError:
            raise ConfigError(f"{key} must be an integer") from None
    return raw.strip()


def load() -> Config:
    """Read the saved config, falling back to defaults when none exists."""
    path = paths.config_path()
    if not path.exists():
        return Config()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ConfigError(f"Could not read config at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Config at {path} is not a JSON object")
    return Config.from_dict(data)


def save(config: Config) -> Path:
    """Write the config atomically so an interrupted write cannot corrupt it."""
    path = paths.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path
