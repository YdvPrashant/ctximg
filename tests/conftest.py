"""Shared fixtures.

Every test runs against a throwaway CTXIMG_HOME so nothing touches the real
config or index in %APPDATA% / %LOCALAPPDATA%.
"""

from __future__ import annotations

import pytest

from ctximg import paths


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Redirect config and data directories into the test's tmp_path."""
    home = tmp_path / "ctximg_home"
    monkeypatch.setenv(paths.HOME_ENV, str(home))
    return home


@pytest.fixture
def gallery(tmp_path):
    """A small folder tree with image-like and non-image files."""
    root = tmp_path / "gallery"
    (root / "trip" / "day1").mkdir(parents=True)
    (root / ".hidden").mkdir()
    (root / "node_modules").mkdir()

    files = [
        root / "a.jpg",
        root / "b.PNG",
        root / "notes.txt",
        root / "trip" / "c.jpeg",
        root / "trip" / "day1" / "d.heic",
        root / ".hidden" / "secret.jpg",
        root / "node_modules" / "vendor.jpg",
    ]
    for i, f in enumerate(files):
        f.write_bytes(b"x" * (10 + i))
    return root
