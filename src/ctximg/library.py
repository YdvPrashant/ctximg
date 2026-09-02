"""Every indexed folder, searched together.

Each folder keeps its own index, so searching several means stacking their
vectors into one matrix. That is only meaningful when the vectors came from the
same model: ViT-B-32 produces 512 numbers per photo and ViT-L-14 produces 768,
so mixing them is not merely inaccurate, it has no valid shape. Folders built by
a different model are therefore listed and reported, never quietly stacked.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config as config_mod
from . import workspace
from .indexer import Progress, run_index
from .search import Hit, relative_match, top_k
from .store import Store


@dataclass
class FolderEntry:
    """One indexed folder, as the app needs to show it."""

    key: str
    root: Path
    photos: int
    model: str | None
    on_disk: bool
    compatible: bool
    selected: bool

    @property
    def name(self) -> str:
        return self.root.name or str(self.root)

    @property
    def state(self) -> str:
        if not self.on_disk:
            return "missing"
        if not self.compatible:
            return "needs_reindex"
        return "ready"

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "path": str(self.root),
            "name": self.name,
            "photos": self.photos,
            "model": self.model,
            "state": self.state,
            "selected": self.selected,
        }


@dataclass
class _Loaded:
    """One folder's vectors, plus the version they were read at."""

    ids: np.ndarray
    matrix: np.ndarray
    version: int


@dataclass
class JobStatus:
    """A background index run, reported to the page while it works."""

    key: str = ""
    path: str = ""
    progress: Progress = field(default_factory=Progress)
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "path": self.path,
            "running": self.progress.running,
            "phase": self.progress.phase,
            "done": self.progress.done,
            "total": self.progress.total,
            "message": self.progress.message,
            "rate": round(self.progress.rate, 1),
            "eta": round(self.progress.eta),
            "elapsed": round(self.progress.elapsed),
            "downloaded": self.progress.downloaded,
            "cancelled": self.progress.cancelled,
            "error": self.error,
        }


class Library:
    """All indexed folders; ranks across the selected, compatible ones."""

    def __init__(self, app):
        self.app = app
        self._cache: dict[str, _Loaded] = {}
        self._matrix: np.ndarray | None = None
        self._owners: list[tuple[str, int]] = []   # row -> (folder key, image id)
        self._signature: tuple | None = None
        self._lock = threading.Lock()
        self.job = JobStatus()

    # --- folders -----------------------------------------------------------

    @property
    def model_id(self) -> str | None:
        """The model the app is currently running, or None if not loaded yet."""
        embedder = self.app.peek_embedder()
        return embedder.model_id if embedder is not None else None

    def folders(self) -> list[FolderEntry]:
        selected = set(self.app.config.selected)
        current = self.model_id
        entries = []
        for info in workspace.known():
            entries.append(
                FolderEntry(
                    key=info.key,
                    root=info.root,
                    photos=info.photos,
                    model=info.model,
                    on_disk=info.on_disk,
                    # Before the model has loaded nothing is known to clash, so
                    # folders read as usable rather than briefly all-broken.
                    compatible=current is None or info.model == current,
                    selected=info.key in selected or not selected,
                )
            )
        return entries

    def searchable(self) -> list[FolderEntry]:
        return [f for f in self.folders() if f.selected and f.state == "ready"]

    def set_selected(self, keys) -> list[FolderEntry]:
        self.app.config.selected = sorted(set(keys))
        config_mod.save(self.app.config)
        return self.folders()

    # --- the stacked matrix ------------------------------------------------

    def _signature_now(self, chosen: list[FolderEntry]) -> tuple:
        """Identity of the current view: which folders, at which versions.

        Includes each folder's vector version, so an index run from the CLI or
        another tab invalidates the stack rather than being missed.
        """
        parts = []
        for entry in chosen:
            space = workspace.Workspace(root=entry.root, key=entry.key)
            try:
                with Store(space.db_path) as store:
                    parts.append((entry.key, store.vectors_version()))
            except Exception:
                parts.append((entry.key, -1))
        return tuple(parts)

    def _rebuild(self, chosen: list[FolderEntry]) -> None:
        matrices, owners = [], []
        for entry in chosen:
            space = workspace.Workspace(root=entry.root, key=entry.key)
            with Store(space.db_path) as store:
                version = store.vectors_version()
                cached = self._cache.get(entry.key)
                if cached is None or cached.version != version:
                    ids, matrix = store.load_matrix()
                    cached = _Loaded(ids=ids, matrix=matrix, version=version)
                    self._cache[entry.key] = cached
            if cached.matrix.shape[0] == 0:
                continue
            matrices.append(cached.matrix)
            owners.extend((entry.key, int(i)) for i in cached.ids)

        if not matrices:
            self._matrix, self._owners = None, []
            return

        widths = {m.shape[1] for m in matrices}
        if len(widths) > 1:
            # Should be unreachable: compatibility filtering runs first.
            raise RuntimeError(f"cannot stack vectors of differing widths: {widths}")
        self._matrix = np.vstack(matrices)
        self._owners = owners

    def _ensure(self) -> None:
        chosen = self.searchable()
        signature = self._signature_now(chosen)
        if signature != self._signature:
            self._rebuild(chosen)
            self._signature = signature

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()
            self._matrix = None
            self._owners = []
            self._signature = None

    @property
    def size(self) -> int:
        with self._lock:
            self._ensure()
            return 0 if self._matrix is None else int(self._matrix.shape[0])

    # --- search ------------------------------------------------------------

    def search(self, query: str, k: int = 60, ensemble: bool = True) -> list[Hit]:
        vector = self.app.embedder.encode_query(query, ensemble=ensemble)
        return self._rank(vector, k)

    def similar(self, key: str, image_id: int, k: int = 60) -> list[Hit]:
        info = workspace.find(key)
        if info is None:
            return []
        with Store(info.workspace().db_path) as store:
            vector = store.get_vector(image_id)
        if vector is None:
            return []
        return self._rank(vector, k + 1, exclude=(key, image_id))[:k]

    def _rank(self, vector, k: int, exclude=None) -> list[Hit]:
        with self._lock:
            self._ensure()
            if self._matrix is None:
                return []
            rows, scores = top_k(self._matrix, vector, k)
            owners = [self._owners[int(r)] for r in rows]

        if exclude is not None:
            keep = [i for i, owner in enumerate(owners) if owner != exclude]
            owners = [owners[i] for i in keep]
            scores = scores[keep]
        if not owners:
            return []

        # One lookup per folder rather than one per hit.
        by_folder: dict[str, list[int]] = {}
        for folder_key, image_id in owners:
            by_folder.setdefault(folder_key, []).append(image_id)

        meta: dict[tuple[str, int], dict] = {}
        roots: dict[str, Path] = {}
        for folder_key, image_ids in by_folder.items():
            info = workspace.find(folder_key)
            if info is None:
                continue
            roots[folder_key] = info.root
            with Store(info.workspace().db_path) as store:
                for image_id, row in store.get_images(image_ids).items():
                    meta[(folder_key, image_id)] = row

        matches = relative_match(scores)
        hits = []
        for rank, ((folder_key, image_id), score, match) in enumerate(
            zip(owners, scores, matches), 1
        ):
            row = meta.get((folder_key, image_id), {})
            hit = Hit(
                image_id=image_id,
                score=float(score),
                match=float(match),
                rank=rank,
                path=row.get("path", ""),
                width=row.get("width"),
                height=row.get("height"),
                taken_at=row.get("taken_at"),
            )
            hit.folder_key = folder_key
            hit.folder_name = roots.get(folder_key, Path("")).name
            hits.append(hit)
        return hits

    # --- estimating --------------------------------------------------------

    def expected_rate(self, device: str | None = None) -> float:
        """Images per second to expect: measured if we have it, else reference."""
        from .embedder import describe_tiers

        embedder = self.app.peek_embedder()
        device = device or (embedder.device if embedder else self.app._likely_device())
        for tier in describe_tiers(device, self.app.config.rates):
            if tier["key"] == self.app.config.tier:
                return tier["rate"]
        return 1.0

    def estimate(self, folder) -> dict:
        """How much work a folder is, before anyone commits to doing it."""
        from .scan import iter_images

        root = Path(folder).expanduser()
        if not root.is_dir():
            return {"ok": False, "message": f"No such folder: {root}"}

        settings = self.app.config
        photos = 0
        total_bytes = 0
        for entry in iter_images([root], settings.extensions, settings.ignore_dirs):
            photos += 1
            total_bytes += entry.size

        space = workspace.for_folder(root)
        already = workspace.find(space.key)
        rate = self.expected_rate()
        return {
            "ok": True,
            "path": str(space.root),
            "name": space.root.name,
            "photos": photos,
            "bytes": total_bytes,
            "already_indexed": already.photos if already else 0,
            "rate": round(rate, 1),
            "eta": round(photos / rate) if rate > 0 else 0,
        }

    # --- indexing ----------------------------------------------------------

    def start_index(self, folder, rebuild: bool = False) -> tuple[bool, str]:
        """Index a folder on a worker thread. Returns (started, message)."""
        with self._lock:
            if self.job.progress.running:
                return False, "Another folder is being indexed."

        root = Path(folder).expanduser()
        if not root.is_dir():
            return False, f"No such folder: {root}"
        space = workspace.for_folder(root)

        self.job = JobStatus(key=space.key, path=str(space.root))
        self.job.progress.running = True
        self.job.progress.phase = "starting"

        def worker():
            store = None
            try:
                # Loading the model is the longest blocking step and, the first
                # time, is a multi-gigabyte download. Left unlabelled it looks
                # exactly like a hang, so say so and watch the bytes arrive.
                if self.app.peek_embedder() is None:
                    self.job.progress.phase = "loading model"
                    watcher = threading.Thread(
                        target=self._watch_download, daemon=True,
                        name="ctximg-download",
                    )
                    watcher.start()
                embedder = self.app.embedder
                if self.job.progress.cancelled:
                    self.job.progress.message = "Stopped before indexing began."
                    return
                self.job.progress.phase = "scanning"
                store = Store(space.db_path)
                store.set_meta(workspace.ROOT_KEY, str(space.root))
                store.commit()
                outcome = run_index(
                    store, embedder, self.app.config, space.thumbs_dir,
                    [space.root], rebuild=rebuild, progress=self.job.progress,
                )
                # Estimates should get truthful about this machine, not stay
                # anchored to a number measured on someone else's.
                if outcome.embedded >= 8 and outcome.rate > 0:
                    embedder = self.app.embedder
                    self.app.config.remember_rate(
                        embedder.model_id, embedder.device, outcome.rate
                    )
                    config_mod.save(self.app.config)
            except Exception as exc:
                self.job.error = f"{type(exc).__name__}: {exc}"
                self.job.progress.running = False
                self.job.progress.phase = "error"
            finally:
                if store is not None:
                    store.close()
                # Whatever happened - finished, cancelled, or raised - the job
                # must stop claiming to be running. A stuck flag is what makes
                # the app look hung with no way out.
                self.job.progress.running = False
                if self.job.progress.phase not in ("error",):
                    self.job.progress.phase = "idle"
                self.invalidate()

        threading.Thread(target=worker, daemon=True, name="ctximg-index").start()
        return True, f"Indexing {space.root}"

    def _watch_download(self) -> None:
        """Poll how much of the model has arrived, while it is arriving."""
        from . import sysinfo

        while self.job.progress.phase == "loading model":
            try:
                self.job.progress.downloaded = sysinfo.download_bytes()
            except Exception:
                pass
            if not self.job.progress.running:
                return
            threading.Event().wait(1.0)
        self.job.progress.downloaded = 0

    def stop(self) -> tuple[bool, str]:
        """Ask a running index to stop at the next safe point.

        Everything already embedded stays: the run commits per batch, so
        stopping costs at most the batch in flight.
        """
        if not self.job.progress.running:
            return False, "Nothing is running."
        self.job.progress.cancel()
        return True, "Stopping after the current batch\u2026"

    def forget(self, key: str) -> bool:
        """Delete a folder's index. Its photos are left exactly where they are."""
        info = workspace.find(key)
        if info is None:
            return False
        removed = workspace.forget(info)
        self.app.config.selected = [
            k for k in self.app.config.selected if k != key
        ]
        config_mod.save(self.app.config)
        self.invalidate()
        return removed
