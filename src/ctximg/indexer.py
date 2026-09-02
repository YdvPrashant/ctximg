"""Indexing: scan the gallery, embed what changed, commit as you go.

Decoding runs on a thread pool while the model encodes the previous batch, and
every batch is committed before the next starts - so Ctrl+C costs one batch,
never the whole run.
"""

from __future__ import annotations

import itertools
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

from . import images as imglib
from .config import Config
from .scan import iter_images
from .store import Store, SyncResult


class FolderUnavailable(RuntimeError):
    """A folder to be indexed cannot be read right now.

    Raised instead of syncing, because an unreadable folder is indistinguishable
    from an emptied one and the index would be thrown away.
    """

    def __init__(self, folders):
        self.folders = [str(f) for f in folders]
        listed = "\n  ".join(self.folders)
        super().__init__(
            f"Cannot read this folder right now, so the index was left alone:\n"
            f"  {listed}\n"
            f"Reconnect the drive or restore the folder, then index again."
        )


@dataclass
class Prepared:
    image_id: int
    tensor: object | None = None
    meta: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class IndexResult:
    sync: SyncResult = field(default_factory=SyncResult)
    embedded: int = 0
    failed: int = 0
    interrupted: bool = False
    total_pending: int = 0
    rate: float = 0.0        # images/second actually achieved

    def summary(self) -> str:
        parts = [
            f"{self.sync.added} new",
            f"{self.sync.updated} changed",
            f"{self.sync.removed} removed",
            f"{self.embedded} embedded",
        ]
        if self.failed:
            parts.append(f"{self.failed} failed")
        if self.interrupted:
            parts.append("interrupted (re-run to resume)")
        return ", ".join(parts)


@dataclass
class Progress:
    """Live counters, polled by the web UI while indexing runs."""

    running: bool = False
    phase: str = "idle"
    done: int = 0
    total: int = 0
    message: str = ""
    rate: float = 0.0        # images/second, measured over this run
    eta: float = 0.0         # seconds of work left, at that rate
    elapsed: float = 0.0
    started_at: float = 0.0
    cancelled: bool = False
    downloaded: int = 0      # bytes pulled so far, while fetching a model

    def cancel(self) -> None:
        self.cancelled = True
        self.phase = "stopping"

    def tick(self) -> None:
        """Recompute speed and time remaining from what has happened so far."""
        if not self.started_at:
            return
        self.elapsed = max(time.monotonic() - self.started_at, 1e-6)
        self.rate = self.done / self.elapsed
        remaining = max(self.total - self.done, 0)
        self.eta = remaining / self.rate if self.rate > 0 else 0.0


def _chunks(iterable: Iterable, size: int) -> Iterator[list]:
    it = iter(iterable)
    while chunk := list(itertools.islice(it, size)):
        yield chunk


def _stream_prepared(
    items: list[tuple[int, str]],
    prepare: Callable[[tuple[int, str]], Prepared],
    workers: int,
    window: int,
) -> Iterator[Prepared]:
    """Run prepare() over items with a bounded number in flight, in order.

    The sliding window keeps decode ahead of the model without materialising a
    future per file - a 100k-photo gallery would otherwise queue 100k of them.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = iter(items)
        inflight = deque(
            pool.submit(prepare, item) for item in itertools.islice(pending, window)
        )
        while inflight:
            future = inflight.popleft()
            nxt = next(pending, None)
            if nxt is not None:
                inflight.append(pool.submit(prepare, nxt))
            yield future.result()


def run_index(
    store: Store,
    embedder,
    config: Config,
    thumbs_root: Path,
    roots: Iterable[str | os.PathLike[str]],
    rebuild: bool = False,
    progress: Progress | None = None,
    on_batch: Callable[[int, int], None] | None = None,
) -> IndexResult:
    """Bring the index up to date with the given folders.

    Roots are passed in rather than read from config: an index belongs to the
    folder it was built for, and config now carries only settings.
    """
    progress = progress or Progress()
    result = IndexResult()

    progress.running = True
    progress.phase = "scanning"
    progress.message = "Walking gallery folders"
    try:
        if rebuild:
            store.rebind_model(embedder.model_id, embedder.dim)
        else:
            store.ensure_model(embedder.model_id, embedder.dim)

        wanted = [Path(r) for r in roots]
        unreachable = [r for r in wanted if not r.is_dir()]
        if wanted and unreachable:
            # sync_files reconciles the index against what the scan finds, so a
            # folder that cannot be read looks exactly like a folder whose
            # every photo was deleted - and the index would be destroyed. An
            # unplugged drive must cost nothing.
            raise FolderUnavailable(unreachable)

        entries = iter_images(wanted, config.extensions, config.ignore_dirs)
        result.sync = store.sync_files(entries)

        pending = store.pending()
        result.total_pending = len(pending)
        progress.phase = "embedding"
        progress.total = len(pending)
        progress.done = 0
        progress.started_at = time.monotonic()
        if not pending:
            progress.message = "Everything is already indexed"
            return result

        batch_size = config.batch_size or _auto_batch(embedder)
        workers = min(8, (os.cpu_count() or 4))
        thumb_size = config.thumb_size

        def prepare(item: tuple[int, str]) -> Prepared:
            image_id, path = item
            try:
                image, meta = imglib.open_upright(path)
            except Exception as exc:
                return Prepared(image_id, error=f"{type(exc).__name__}: {exc}")
            try:
                imglib.write_thumb(
                    image, imglib.thumb_path(thumbs_root, image_id), thumb_size
                )
            except Exception:
                pass  # a missing thumbnail must not cost us the embedding
            try:
                tensor = embedder.preprocess(image)
            except Exception as exc:
                return Prepared(image_id, error=f"preprocess: {exc}")
            return Prepared(image_id, tensor=tensor, meta=meta)

        stream = _stream_prepared(pending, prepare, workers, batch_size * 3)
        batch: list[Prepared] = []

        def flush() -> None:
            if not batch:
                return
            vectors = embedder.encode_images([p.tensor for p in batch])
            store.save_vectors(
                [(p.image_id, vectors[i], p.meta) for i, p in enumerate(batch)]
            )
            result.embedded += len(batch)
            progress.done = result.embedded + result.failed
            progress.tick()
            if on_batch:
                on_batch(progress.done, progress.total)
            batch.clear()

        try:
            for prepared in stream:
                if progress.cancelled:
                    flush()
                    store.commit()
                    result.interrupted = True
                    break
                if prepared.error:
                    store.mark_error(prepared.image_id, prepared.error)
                    result.failed += 1
                    progress.done = result.embedded + result.failed
                    progress.tick()
                    continue
                batch.append(prepared)
                if len(batch) >= batch_size:
                    flush()
            flush()
        except KeyboardInterrupt:
            flush()
            store.commit()
            result.interrupted = True

        result.rate = progress.rate
        progress.message = ("Stopped. " if result.interrupted else "") + result.summary()
        return result
    finally:
        store.commit()
        progress.running = False
        progress.phase = "idle"


def _auto_batch(embedder) -> int:
    from .embedder import default_batch_size

    return default_batch_size(getattr(embedder, "device", "cpu"))
