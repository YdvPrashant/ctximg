"""Per-folder indexes, so any folder can be searched without registering it.

Each folder gets its own index under %LOCALAPPDATA%\\ctximg\\folders, keyed by a
hash of its absolute path. The photo folder itself is never written to, which
means read-only media and network shares work the same as a local disk.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import paths

ROOT_KEY = "root_path"


def _normalise(root: Path) -> str:
    """The string a folder's key is derived from.

    Windows paths are case-insensitive, so C:\\Photos and c:\\photos are the
    same folder and must land on the same index.
    """
    text = str(root)
    return text.lower() if os.name == "nt" else text


def _slug(name: str) -> str:
    """A readable, filesystem-safe fragment of the folder name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return (cleaned[:32] or "folder").lower()


@dataclass(frozen=True)
class Workspace:
    """Where one folder's index lives."""

    root: Path
    key: str

    @property
    def dir(self) -> Path:
        return paths.data_dir() / "folders" / f"{_slug(self.root.name)}-{self.key}"

    @property
    def db_path(self) -> Path:
        return self.dir / "index.db"

    @property
    def thumbs_dir(self) -> Path:
        return self.dir / "thumbs"

    def exists(self) -> bool:
        """True when this folder has been indexed before."""
        return self.db_path.exists()


def for_folder(folder: str | os.PathLike[str]) -> Workspace:
    """Resolve the workspace for a folder, without creating anything."""
    root = Path(folder).expanduser().resolve()
    digest = hashlib.sha256(_normalise(root).encode("utf-8")).hexdigest()[:12]
    return Workspace(root=root, key=digest)


@dataclass(frozen=True)
class FolderInfo:
    """What an index says about itself, read without opening it for writing."""

    key: str
    root: Path
    dir: Path
    model: str | None
    dim: int | None
    photos: int

    @property
    def on_disk(self) -> bool:
        return self.root.is_dir()

    def workspace(self) -> "Workspace":
        return Workspace(root=self.root, key=self.key)


def describe(entry: Path) -> FolderInfo | None:
    """Read one workspace directory, or None if it holds no usable index."""
    db = entry / "index.db"
    if not db.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            photos = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error:
        return None

    root = meta.get(ROOT_KEY)
    if not root:
        return None
    dim = meta.get("embed_dim")
    return FolderInfo(
        key=entry.name.rsplit("-", 1)[-1],
        root=Path(root),
        dir=entry,
        model=meta.get("model_id"),
        dim=int(dim) if dim else None,
        photos=photos,
    )


def known(include_empty: bool = False) -> list[FolderInfo]:
    """Every folder indexed so far, newest first.

    Workspaces holding no vectors are skipped by default: an index that was
    created and then abandoned is not a folder the user has indexed, and
    listing it as one is a lie.
    """
    base = paths.data_dir() / "folders"
    if not base.is_dir():
        return []

    found: list[tuple[float, FolderInfo]] = []
    for entry in sorted(base.iterdir()):
        info = describe(entry)
        if info is None or (info.photos == 0 and not include_empty):
            continue
        found.append(((entry / "index.db").stat().st_mtime, info))
    found.sort(key=lambda pair: pair[0], reverse=True)
    return [info for _, info in found]


def find(key: str) -> FolderInfo | None:
    """Look up one indexed folder by its key."""
    return next((info for info in known(include_empty=True) if info.key == key), None)


def forget(info: FolderInfo) -> bool:
    """Delete one folder's index. The photos themselves are never touched.

    This is the only place the app removes a directory tree, so it refuses to
    act on anything that is not inside our own index store - a corrupted or
    hand-edited record must not be able to point rmtree at a photo folder.
    """
    base = (paths.data_dir() / "folders").resolve()
    target = info.dir.resolve()
    if base not in target.parents:
        raise ValueError(f"refusing to delete outside the index store: {target}")
    if not target.is_dir():
        return False
    shutil.rmtree(target)
    return True
