"""SQLite index: file metadata plus one embedding vector per image.

Vectors are stored L2-normalised as float16 blobs, which halves the file size
at a cosine error well under the gap between adjacent search results.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS images (
    id         INTEGER PRIMARY KEY,
    path       TEXT UNIQUE NOT NULL,
    size       INTEGER NOT NULL,
    mtime      REAL NOT NULL,
    width      INTEGER,
    height     INTEGER,
    taken_at   TEXT,
    embedded   INTEGER NOT NULL DEFAULT 0,
    error      TEXT,
    indexed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_images_pending ON images(embedded, error);
CREATE TABLE IF NOT EXISTS vectors (
    image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
    vec      BLOB NOT NULL
);
"""


class ModelMismatch(RuntimeError):
    """The index holds vectors from a different model than the one requested.

    Vectors from two models occupy unrelated spaces; comparing them returns
    plausible-looking nonsense, so this is a hard error rather than a warning.
    """

    def __init__(self, stored: str, requested: str):
        self.stored = stored
        self.requested = requested
        super().__init__(
            f"Index was built with model {stored!r} but {requested!r} is configured. "
            f"Run 'ctximg index --rebuild' to re-embed everything with the new model, "
            f"or go back with 'ctximg config set model {stored}'."
        )


@dataclass
class SyncResult:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0

    @property
    def changed(self) -> int:
        return self.added + self.updated + self.removed


@dataclass
class Stats:
    total: int = 0
    embedded: int = 0
    failed: int = 0
    pending: int = 0
    model: str | None = None
    dim: int | None = None
    db_bytes: int = 0


class _LockedConnection:
    """Serialises statements so the web server can read while indexing writes.

    Every caller goes through execute/executescript/commit, so locking here
    covers the whole surface without threading a lock through each method.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._lock = threading.RLock()

    def execute(self, sql, params=()):
        with self._lock:
            return self._conn.execute(sql, params)

    def executescript(self, sql):
        with self._lock:
            return self._conn.executescript(sql)

    def commit(self):
        with self._lock:
            return self._conn.commit()

    def close(self):
        with self._lock:
            return self._conn.close()

    def __setattr__(self, name, value):
        if name in ("_conn", "_lock"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._conn, name, value)


class Store:
    """Owns one SQLite connection; safe to read from several threads."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = _LockedConnection(
            sqlite3.connect(str(self.path), check_same_thread=False)
        )
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(_SCHEMA)
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", SCHEMA_VERSION)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- meta --------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )

    def ensure_model(self, model_id: str, dim: int) -> None:
        """Bind the index to a model, or raise if it is already bound elsewhere."""
        stored = self.get_meta("model_id")
        if stored is None:
            self.set_meta("model_id", model_id)
            self.set_meta("embed_dim", str(dim))
            self.db.commit()
            return
        if stored != model_id:
            raise ModelMismatch(stored, model_id)
        if self.get_meta("embed_dim") != str(dim):
            self.set_meta("embed_dim", str(dim))
            self.db.commit()

    def embed_dim(self) -> int | None:
        raw = self.get_meta("embed_dim")
        return int(raw) if raw else None

    def vectors_version(self) -> int:
        """Counter bumped on every change to the vector set.

        A running server compares this against the version it loaded, so an
        index run from a separate process is picked up instead of the server
        serving a stale matrix. One indexed row read, so it costs nothing to
        check on every query.
        """
        raw = self.get_meta("vectors_version")
        return int(raw) if raw else 0

    def _bump_version(self) -> None:
        self.set_meta("vectors_version", str(self.vectors_version() + 1))

    def rebind_model(self, model_id: str, dim: int) -> None:
        """Drop every vector and bind the index to a new model."""
        self.db.execute("DELETE FROM vectors")
        self.db.execute("UPDATE images SET embedded = 0, error = NULL, indexed_at = NULL")
        self.set_meta("model_id", model_id)
        self.set_meta("embed_dim", str(dim))
        self._bump_version()
        self.db.commit()

    # --- file synchronisation ---------------------------------------------

    def _plan(self, entries: Iterable) -> tuple[list, list, list, int]:
        """Work out what changed on disk, touching nothing.

        Returns (new, changed, gone, unchanged_count), where new and changed
        carry entries and gone carries paths.
        """
        existing = {
            row["path"]: (row["id"], row["size"], row["mtime"])
            for row in self.db.execute("SELECT id, path, size, mtime FROM images")
        }
        new, changed = [], []
        seen: set[str] = set()
        unchanged = 0

        for entry in entries:
            seen.add(entry.path)
            prior = existing.get(entry.path)
            if prior is None:
                new.append(entry)
            elif prior[1] != entry.size or abs(prior[2] - entry.mtime) > 1e-6:
                changed.append((prior[0], entry))
            else:
                unchanged += 1

        gone = [path for path in existing if path not in seen]
        return new, changed, gone, unchanged

    def preview_sync(self, entries: Iterable) -> SyncResult:
        """What sync_files would do, without doing it.

        Lets a folder be reported as up to date or drifted before anything is
        written - opening a folder to look at it must not modify its index.
        """
        new, changed, gone, unchanged = self._plan(entries)
        return SyncResult(
            added=len(new), updated=len(changed), removed=len(gone), unchanged=unchanged
        )

    def sync_files(self, entries: Iterable) -> SyncResult:
        """Reconcile the index against what is on disk right now.

        New files are queued for embedding, changed files are re-queued and
        lose their stale vector, and vanished files are dropped so results can
        never point at a path that is gone.
        """
        new, changed, gone, unchanged = self._plan(entries)
        result = SyncResult(
            added=len(new), updated=len(changed), removed=len(gone), unchanged=unchanged
        )
        now = time.time()

        for entry in new:
            self.db.execute(
                "INSERT INTO images(path, size, mtime, embedded) VALUES(?, ?, ?, 0)",
                (entry.path, entry.size, entry.mtime),
            )
        for image_id, entry in changed:
            self.db.execute(
                "UPDATE images SET size = ?, mtime = ?, embedded = 0, error = NULL, "
                "indexed_at = ? WHERE id = ?",
                (entry.size, entry.mtime, now, image_id),
            )
            self.db.execute("DELETE FROM vectors WHERE image_id = ?", (image_id,))
        for start in range(0, len(gone), 500):
            chunk = gone[start:start + 500]
            placeholders = ",".join("?" * len(chunk))
            self.db.execute(f"DELETE FROM images WHERE path IN ({placeholders})", chunk)

        if result.changed:
            self._bump_version()
        self.db.commit()
        return result

    # --- embedding queue ---------------------------------------------------

    def pending(self, limit: int | None = None) -> list[tuple[int, str]]:
        """Images still needing a vector, oldest first. Errored files are excluded."""
        sql = "SELECT id, path FROM images WHERE embedded = 0 AND error IS NULL ORDER BY id"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return [(r["id"], r["path"]) for r in self.db.execute(sql)]

    def pending_count(self) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) AS n FROM images WHERE embedded = 0 AND error IS NULL"
        ).fetchone()
        return row["n"]

    def save_vectors(self, items: Sequence[tuple[int, np.ndarray, dict]]) -> None:
        """Persist a batch of vectors and their image metadata, then commit.

        Committing per batch is what makes indexing resumable: Ctrl+C costs at
        most one batch of work.
        """
        now = time.time()
        for image_id, vec, meta in items:
            arr = np.asarray(vec, dtype=np.float32).ravel()
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr = arr / norm
            self.db.execute(
                "INSERT INTO vectors(image_id, vec) VALUES(?, ?) "
                "ON CONFLICT(image_id) DO UPDATE SET vec = excluded.vec",
                (image_id, arr.astype(np.float16).tobytes()),
            )
            self.db.execute(
                "UPDATE images SET embedded = 1, error = NULL, indexed_at = ?, "
                "width = ?, height = ?, taken_at = ? WHERE id = ?",
                (now, meta.get("width"), meta.get("height"), meta.get("taken_at"), image_id),
            )
        if items:
            self._bump_version()
        self.db.commit()

    def mark_error(self, image_id: int, message: str) -> None:
        """Record a permanent failure so the file is not retried on every run."""
        self.db.execute(
            "UPDATE images SET error = ?, embedded = 0, indexed_at = ? WHERE id = ?",
            (message[:500], time.time(), image_id),
        )

    def commit(self) -> None:
        self.db.commit()

    # --- reading -----------------------------------------------------------

    def load_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (image_ids, matrix) with unit-norm float32 rows.

        Blobs are concatenated once and reinterpreted in a single call rather
        than decoded row by row.
        """
        dim = self.embed_dim()
        rows = self.db.execute(
            "SELECT image_id, vec FROM vectors ORDER BY image_id"
        ).fetchall()
        if not rows or dim is None:
            return np.empty(0, dtype=np.int64), np.empty((0, dim or 0), dtype=np.float32)

        ids = np.fromiter((r["image_id"] for r in rows), dtype=np.int64, count=len(rows))
        raw = b"".join(r["vec"] for r in rows)
        mat = np.frombuffer(raw, dtype=np.float16).reshape(len(rows), dim).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        np.divide(mat, np.maximum(norms, 1e-12), out=mat)
        return ids, mat

    def get_vector(self, image_id: int) -> np.ndarray | None:
        row = self.db.execute(
            "SELECT vec FROM vectors WHERE image_id = ?", (image_id,)
        ).fetchone()
        if row is None:
            return None
        vec = np.frombuffer(row["vec"], dtype=np.float16).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def get_images(self, image_ids: Sequence[int]) -> dict[int, dict]:
        if not image_ids:
            return {}
        placeholders = ",".join("?" * len(image_ids))
        rows = self.db.execute(
            f"SELECT id, path, size, width, height, taken_at FROM images "
            f"WHERE id IN ({placeholders})",
            list(image_ids),
        )
        return {r["id"]: dict(r) for r in rows}

    def get_image(self, image_id: int) -> dict | None:
        row = self.db.execute(
            "SELECT id, path, size, width, height, taken_at FROM images WHERE id = ?",
            (image_id,),
        ).fetchone()
        return dict(row) if row else None

    def failures(self, limit: int = 20) -> list[tuple[str, str]]:
        rows = self.db.execute(
            "SELECT path, error FROM images WHERE error IS NOT NULL LIMIT ?", (limit,)
        )
        return [(r["path"], r["error"]) for r in rows]

    def stats(self) -> Stats:
        row = self.db.execute(
            "SELECT COUNT(*) AS total,"
            " SUM(CASE WHEN embedded = 1 THEN 1 ELSE 0 END) AS embedded,"
            " SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS failed"
            " FROM images"
        ).fetchone()
        total = row["total"] or 0
        embedded = row["embedded"] or 0
        failed = row["failed"] or 0
        return Stats(
            total=total,
            embedded=embedded,
            failed=failed,
            pending=total - embedded - failed,
            model=self.get_meta("model_id"),
            dim=self.embed_dim(),
            db_bytes=self.path.stat().st_size if self.path.exists() else 0,
        )
