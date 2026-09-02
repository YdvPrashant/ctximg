"""Retrieval: cosine similarity over the whole gallery, ranked.

There is no keyword index, no filter, no threshold and no boolean parsing here.
Every photo is scored against the query vector and the best ones float up -
that is the entire retrieval model, deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Hit:
    image_id: int
    score: float          # raw cosine similarity, typically ~0.15-0.35
    match: float          # 0-1, relative within this result set, for display
    rank: int
    path: str = ""
    width: int | None = None
    height: int | None = None
    taken_at: str | None = None
    # Which folder's index this came from. Results can span folders, so an
    # image id alone no longer identifies a photo.
    folder_key: str = ""
    folder_name: str = ""

    def as_dict(self) -> dict:
        return {
            "id": self.image_id,
            "folder": self.folder_key,
            "folder_name": self.folder_name,
            "score": round(self.score, 4),
            "match": round(self.match, 4),
            "rank": self.rank,
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "taken_at": self.taken_at,
        }


def top_k(matrix: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (row_indices, scores) for the k best rows, best first.

    Rows and query are unit vectors, so the dot product is cosine similarity.
    argpartition finds the k best in linear time, and only those k get sorted.
    """
    if matrix.shape[0] == 0 or k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    scores = matrix @ np.asarray(query, dtype=np.float32).ravel()
    k = min(k, scores.shape[0])
    if k == scores.shape[0]:
        order = np.argsort(-scores)
    else:
        head = np.argpartition(-scores, k - 1)[:k]
        order = head[np.argsort(-scores[head])]
    return order, scores[order]


def relative_match(scores: np.ndarray) -> np.ndarray:
    """Rescale a result set to 0-1.

    Raw cosine scores sit in a narrow band that reads as meaningless to a
    human; the spread within the returned set is what actually carries
    information about how much better the top hit is than the tail.
    """
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)


class SearchIndex:
    """Holds the vector matrix in memory and answers queries against it."""

    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder
        self._ids: np.ndarray | None = None
        self._matrix: np.ndarray | None = None
        self._version: int | None = None

    def load(self, force: bool = False) -> int:
        """Read vectors into memory. Returns how many are loaded."""
        if force or self._matrix is None:
            self._version = self.store.vectors_version()
            self._ids, self._matrix = self.store.load_matrix()
        return int(self._ids.shape[0])

    def refresh_if_stale(self) -> bool:
        """Reload when another process has changed the vectors.

        Without this, a server started before `ctximg index` would keep
        answering from the matrix it read at startup and silently omit every
        newly indexed photo.
        """
        if self._matrix is None:
            self.load()
            return True
        if self.store.vectors_version() != self._version:
            self.load(force=True)
            return True
        return False

    def invalidate(self) -> None:
        """Drop the cached matrix so the next query picks up new vectors."""
        self._ids = None
        self._matrix = None
        self._version = None

    @property
    def size(self) -> int:
        self.refresh_if_stale()
        return int(self._ids.shape[0])

    def search(self, query: str, k: int = 60, ensemble: bool = True) -> list[Hit]:
        """Rank the gallery against a natural-language query."""
        self.refresh_if_stale()
        vector = self.embedder.encode_query(query, ensemble=ensemble)
        return self._rank(vector, k)

    def similar(self, image_id: int, k: int = 60) -> list[Hit]:
        """Rank the gallery against one of its own photos."""
        self.refresh_if_stale()
        vector = self.store.get_vector(image_id)
        if vector is None:
            return []
        return self._rank(vector, k + 1, exclude=image_id)[:k]

    def _rank(self, vector: np.ndarray, k: int, exclude: int | None = None) -> list[Hit]:
        rows, scores = top_k(self._matrix, vector, k)
        if rows.size == 0:
            return []

        ids = self._ids[rows]
        if exclude is not None:
            keep = ids != exclude
            ids, scores = ids[keep], scores[keep]

        matches = relative_match(scores)
        meta = self.store.get_images([int(i) for i in ids])
        hits = []
        for rank, (image_id, score, match) in enumerate(zip(ids, scores, matches), 1):
            info = meta.get(int(image_id), {})
            hits.append(
                Hit(
                    image_id=int(image_id),
                    score=float(score),
                    match=float(match),
                    rank=rank,
                    path=info.get("path", ""),
                    width=info.get("width"),
                    height=info.get("height"),
                    taken_at=info.get("taken_at"),
                )
            )
        return hits
