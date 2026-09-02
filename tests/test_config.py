"""Folders must survive across processes - that is the whole point of the config."""

from __future__ import annotations

import pytest

from ctximg import config, paths


def test_defaults_when_no_file_exists():
    cfg = config.load()
    assert cfg.selected == []
    assert cfg.device == "auto"
    assert cfg.model is None
    assert ".heic" in cfg.extensions
    assert not paths.config_path().exists(), "load() must not create the file"


@pytest.mark.parametrize(
    "key,raw,expected",
    [
        ("model", "ViT-L-14", "ViT-L-14"),
        ("model", "auto", None),
        ("batch_size", "64", 64),
        ("batch_size", "auto", None),
        ("thumb_size", "384", 384),
        ("device", "cpu", "cpu"),
        ("extensions", "jpg, .PNG", [".jpg", ".png"]),
    ],
)
def test_set_value_coercion(key, raw, expected):
    cfg = config.load()
    assert cfg.set_value(key, raw) == expected
    assert getattr(cfg, key) == expected


@pytest.mark.parametrize(
    "key,raw",
    [
        ("device", "gpu"),
        ("thumb_size", "8"),
        ("thumb_size", "big"),
        ("batch_size", "0"),
        ("nonsense", "1"),
        ("folders", "C:\\photos"),
    ],
)
def test_set_value_rejects_bad_input(key, raw):
    cfg = config.load()
    with pytest.raises(config.ConfigError):
        cfg.set_value(key, raw)


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    cfg = config.load()
    cfg.selected = ["abc123"]
    written = config.save(cfg)

    assert written.exists()
    assert list(written.parent.glob("*.tmp")) == []


def test_unknown_keys_in_file_are_ignored(tmp_path):
    path = paths.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"selected": [], "from_a_future_version": 7}', encoding="utf-8")

    cfg = config.load()
    assert cfg.selected == []


def test_corrupt_config_raises_clearly():
    path = paths.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(config.ConfigError, match="Could not read config"):
        config.load()
