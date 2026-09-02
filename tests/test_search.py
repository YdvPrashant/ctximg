"""Ranking correctness, proved against hand-built vectors - no model download."""

from __future__ import annotations

import numpy as np
import pytest

from ctximg import paths
from ctximg.scan import FileEntry
from ctximg.search import SearchIndex, relative_match, top_k
from ctximg.store import Store


class StubEmbedder:
    """Stands in for CLIP: turns known query strings into known vectors."""

    dim = 3
    model_id = "stub/v1"

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def encode_query(self, query, ensemble=True):
        self.calls.append((query, ensemble))
        vec = np.asarray(self.mapping[query], dtype=np.float32)
        return vec / np.linalg.norm(vec)


# Three photos on the axes, so similarity to each axis is unambiguous.
PHOTOS = {
    "/beach.jpg": [1.0, 0.0, 0.0],
    "/forest.jpg": [0.0, 1.0, 0.0],
    "/city.jpg": [0.0, 0.0, 1.0],
}


@pytest.fixture
def index():
    store = Store(paths.db_path())
    store.ensure_model("stub/v1", 3)
    store.sync_files([FileEntry(p, 10, 1.0) for p in PHOTOS])
    for image_id, path in store.pending():
        store.save_vectors([(image_id, np.array(PHOTOS[path], np.float32), {})])
    embedder = StubEmbedder(
        {
            "beach": [1.0, 0.0, 0.0],
            "forest": [0.0, 1.0, 0.0],
            "leaning beachward": [0.9, 0.4, 0.0],
        }
    )
    yield SearchIndex(store, embedder), store
    store.close()


# --- the pure ranking function -------------------------------------------


def test_top_k_returns_best_first():
    matrix = np.eye(3, dtype=np.float32)
    rows, scores = top_k(matrix, np.array([0.0, 1.0, 0.0], np.float32), 3)
    assert list(rows) == [1, 0, 2] or (rows[0] == 1 and scores[0] == 1.0)
    assert scores[0] == pytest.approx(1.0)
    assert list(scores) == sorted(scores, reverse=True)


def test_top_k_respects_k_and_stays_sorted():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(200, 8)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    query = matrix[7]

    rows, scores = top_k(matrix, query, 5)
    assert len(rows) == 5
    assert rows[0] == 7, "a row must be its own best match"
    assert list(scores) == sorted(scores, reverse=True)

    full = np.sort(matrix @ query)[::-1][:5]
    assert np.allclose(scores, full), "partial selection must match a full sort"


def test_top_k_handles_k_larger_than_corpus_and_empty_corpus():
    matrix = np.eye(2, dtype=np.float32)
    rows, _ = top_k(matrix, np.array([1.0, 0.0], np.float32), 99)
    assert len(rows) == 2

    empty = np.empty((0, 2), dtype=np.float32)
    rows, scores = top_k(empty, np.array([1.0, 0.0], np.float32), 5)
    assert len(rows) == 0 and len(scores) == 0


def test_relative_match_spans_zero_to_one():
    out = relative_match(np.array([0.3, 0.2, 0.1], np.float32))
    assert out[0] == pytest.approx(1.0) and out[-1] == pytest.approx(0.0)


def test_relative_match_handles_identical_scores():
    out = relative_match(np.array([0.25, 0.25], np.float32))
    assert np.allclose(out, 1.0), "no spread must not divide by zero"


# --- the search index -----------------------------------------------------


def test_search_ranks_the_matching_photo_first(index):
    idx, _ = index
    hits = idx.search("forest")
    assert hits[0].path == "/forest.jpg"
    assert hits[0].score == pytest.approx(1.0, abs=1e-3)
    assert hits[0].rank == 1


def test_search_orders_by_similarity_not_by_a_threshold(index):
    idx, _ = index
    hits = idx.search("leaning beachward")
    assert [h.path for h in hits] == ["/beach.jpg", "/forest.jpg", "/city.jpg"]
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))
    assert len(hits) == 3, "every photo is scored; nothing is filtered out"


def test_search_respects_k(index):
    idx, _ = index
    assert len(idx.search("beach", k=1)) == 1


def test_hits_carry_metadata_and_serialise(index):
    idx, _ = index
    payload = idx.search("beach", k=1)[0].as_dict()
    assert payload["id"] > 0
    assert payload["path"] == "/beach.jpg"
    assert 0.0 <= payload["match"] <= 1.0


def test_ensemble_flag_is_passed_through(index):
    idx, _ = index
    idx.search("beach", ensemble=False)
    assert idx.embedder.calls[-1] == ("beach", False)


def test_similar_uses_the_photo_and_excludes_itself(index):
    idx, store = index
    beach_id = next(
        i for i, p in store.get_images([1, 2, 3]).items() if p["path"] == "/beach.jpg"
    )
    hits = idx.similar(beach_id)
    assert beach_id not in [h.image_id for h in hits]
    assert len(hits) == 2


def test_similar_on_an_unknown_id_returns_nothing(index):
    idx, _ = index
    assert idx.similar(9999) == []


def test_search_on_an_empty_index_returns_nothing():
    store = Store(paths.db_path())
    store.ensure_model("stub/v1", 3)
    idx = SearchIndex(store, StubEmbedder({"anything": [1.0, 0.0, 0.0]}))
    assert idx.search("anything") == []
    store.close()


def test_newly_indexed_photos_appear_without_an_explicit_invalidate(index):
    idx, store = index
    assert idx.size == 3

    store.sync_files(
        [FileEntry(p, 10, 1.0) for p in PHOTOS] + [FileEntry("/new.jpg", 10, 1.0)]
    )
    new_id = store.pending()[0][0]
    store.save_vectors([(new_id, np.array([1.0, 0.0, 0.0], np.float32), {})])

    assert idx.size == 4
    assert len(idx.search("beach", k=10)) == 4


def test_index_run_by_another_process_is_picked_up(index):
    """A running server must not keep serving the matrix it read at startup."""
    idx, store = index
    assert idx.size == 3

    # A second connection stands in for a separate `ctximg index` process.
    with Store(paths.db_path()) as other:
        other.sync_files(
            [FileEntry(p, 10, 1.0) for p in PHOTOS] + [FileEntry("/late.jpg", 10, 1.0)]
        )
        late_id = other.pending()[0][0]
        other.save_vectors([(late_id, np.array([0.0, 0.0, 1.0], np.float32), {})])

    assert idx.size == 4, "the new photo must be visible without a restart"
    assert late_id in [h.image_id for h in idx.search("beach", k=10)]


def test_unchanged_index_is_not_reloaded(index):
    idx, _ = index
    idx.load()
    assert idx.refresh_if_stale() is False, "a steady index must not reload per query"


def test_deletions_are_picked_up_too(index):
    idx, store = index
    assert idx.size == 3
    store.sync_files([FileEntry("/beach.jpg", 10, 1.0)])
    assert idx.size == 1
