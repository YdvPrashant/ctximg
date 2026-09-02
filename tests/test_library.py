"""Searching across folders, and refusing to mix incompatible ones."""

from __future__ import annotations

import numpy as np
import pytest

from ctximg import workspace
from ctximg.app import App
from ctximg.config import Config
from ctximg.library import Library
from ctximg.scan import FileEntry
from ctximg.store import Store


class StubEmbedder:
    """Turns known queries into known vectors. Never loads a model."""

    device = "cpu"
    precision = "fp32"
    tier = "fast"
    image_size = 224

    def describe(self):
        return {"model": self.model_id, "device": self.device,
                "precision": self.precision, "dim": self.dim,
                "tier": self.tier, "image_size": self.image_size, "arch": "stub"}

    def __init__(self, model_id="stub/v1", dim=3, mapping=None):
        self.model_id = model_id
        self.dim = dim
        self.mapping = mapping or {}

    def encode_query(self, query, ensemble=True):
        vec = np.asarray(self.mapping[query], dtype=np.float32)
        return vec / np.linalg.norm(vec)


def seed(gallery, photos: dict, model="stub/v1", dim=3):
    """Build a folder's index directly, one vector per named photo."""
    gallery.mkdir(parents=True, exist_ok=True)
    space = workspace.for_folder(gallery)
    with Store(space.db_path) as store:
        store.set_meta(workspace.ROOT_KEY, str(gallery.resolve()))
        store.ensure_model(model, dim)
        store.sync_files(
            [FileEntry(str(gallery / name), 10, 1.0) for name in photos]
        )
        for image_id, path in store.pending():
            name = path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
            store.save_vectors(
                [(image_id, np.asarray(photos[name], np.float32), {})]
            )
    return space


def make_library(mapping=None, model="stub/v1", dim=3):
    app = App(Config())
    app._embedder = StubEmbedder(model_id=model, dim=dim, mapping=mapping or {})
    return Library(app), app


QUERIES = {
    "beach": [1.0, 0.0, 0.0],
    "forest": [0.0, 1.0, 0.0],
    "city": [0.0, 0.0, 1.0],
}


@pytest.fixture
def two_folders(tmp_path):
    seed(tmp_path / "trips", {"beach.jpg": [1.0, 0.0, 0.0]})
    seed(tmp_path / "walks", {"forest.jpg": [0.0, 1.0, 0.0],
                              "city.jpg": [0.0, 0.0, 1.0]})
    return tmp_path


# --- listing --------------------------------------------------------------


def test_lists_every_indexed_folder(two_folders):
    library, _ = make_library(QUERIES)
    names = sorted(f.name for f in library.folders())
    assert names == ["trips", "walks"]
    assert all(f.state == "ready" for f in library.folders())


def test_everything_is_selected_before_you_choose(two_folders):
    """An empty selection means all folders, so a fresh install searches."""
    library, _ = make_library(QUERIES)
    assert all(f.selected for f in library.folders())
    assert len(library.searchable()) == 2


# --- searching across folders --------------------------------------------


def test_search_spans_folders_and_labels_each_hit(two_folders):
    library, _ = make_library(QUERIES)
    hits = library.search("forest", k=10)

    assert len(hits) == 3, "all photos from both folders are scored"
    assert hits[0].path.endswith("forest.jpg")
    assert hits[0].folder_name == "walks"
    assert {h.folder_name for h in hits} == {"trips", "walks"}


def test_ranking_matches_a_single_merged_index(two_folders):
    library, _ = make_library(QUERIES)
    hits = library.search("beach", k=10)

    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0].score == pytest.approx(1.0, abs=1e-3)
    assert hits[0].folder_name == "trips"


def test_deselecting_a_folder_removes_its_photos(two_folders):
    library, _ = make_library(QUERIES)
    walks = next(f for f in library.folders() if f.name == "walks")

    library.set_selected([walks.key])
    hits = library.search("beach", k=10)

    assert {h.folder_name for h in hits} == {"walks"}
    assert not any(h.path.endswith("beach.jpg") for h in hits)


def test_reselecting_brings_it_back(two_folders):
    library, _ = make_library(QUERIES)
    keys = [f.key for f in library.folders()]
    walks = next(f for f in library.folders() if f.name == "walks")

    library.set_selected([walks.key])
    assert library.size == 2
    library.set_selected(keys)
    assert library.size == 3


def test_selection_persists_to_config(two_folders):
    from ctximg import config as config_mod

    library, _ = make_library(QUERIES)
    walks = next(f for f in library.folders() if f.name == "walks")
    library.set_selected([walks.key])

    assert config_mod.load().selected == [walks.key]


# --- the incompatibility that would otherwise corrupt results -------------


def test_a_folder_on_another_model_is_excluded_not_stacked(tmp_path):
    """Different models mean different widths - stacking has no valid shape."""
    seed(tmp_path / "small", {"a.jpg": [1.0, 0.0, 0.0]}, model="stub/v1", dim=3)
    seed(tmp_path / "big", {"b.jpg": [1.0, 0.0, 0.0, 0.0]}, model="other/v2", dim=4)

    library, _ = make_library(QUERIES, model="stub/v1", dim=3)
    states = {f.name: f.state for f in library.folders()}
    assert states == {"small": "ready", "big": "needs_reindex"}

    hits = library.search("beach", k=10)
    assert len(hits) == 1
    assert hits[0].folder_name == "small"


def test_incompatible_folders_are_still_listed(tmp_path):
    seed(tmp_path / "old", {"a.jpg": [1.0, 0.0, 0.0, 0.0]}, model="other/v2", dim=4)
    library, _ = make_library(QUERIES, model="stub/v1", dim=3)

    listed = library.folders()
    assert len(listed) == 1, "it must be visible so it can be fixed"
    assert listed[0].state == "needs_reindex"
    assert library.search("beach", k=10) == []


def test_folders_read_as_usable_before_the_model_loads(two_folders):
    """A cold start must not paint every folder as broken."""
    app = App(Config())
    library = Library(app)
    assert app.peek_embedder() is None
    assert all(f.state == "ready" for f in library.folders())


def test_a_missing_folder_is_flagged_and_skipped(tmp_path):
    gallery = tmp_path / "unplugged"
    seed(gallery, {"a.jpg": [1.0, 0.0, 0.0]})
    for child in gallery.iterdir():
        child.unlink()
    gallery.rmdir()

    library, _ = make_library(QUERIES)
    assert library.folders()[0].state == "missing"
    assert library.search("beach", k=10) == []


# --- staleness ------------------------------------------------------------


def test_new_photos_appear_without_restarting(two_folders):
    library, _ = make_library(QUERIES)
    assert library.size == 3

    space = workspace.for_folder(two_folders / "trips")
    with Store(space.db_path) as store:
        store.sync_files([
            FileEntry(str(two_folders / "trips" / "beach.jpg"), 10, 1.0),
            FileEntry(str(two_folders / "trips" / "late.jpg"), 10, 1.0),
        ])
        new_id = store.pending()[0][0]
        store.save_vectors([(new_id, np.array([0.0, 1.0, 0.0], np.float32), {})])

    assert library.size == 4, "an index written elsewhere must be picked up"


# --- similar and forget ---------------------------------------------------


def test_similar_excludes_the_source_photo(two_folders):
    library, _ = make_library(QUERIES)
    beach = library.search("beach", k=1)[0]

    hits = library.similar(beach.folder_key, beach.image_id, k=10)
    assert (beach.folder_key, beach.image_id) not in [
        (h.folder_key, h.image_id) for h in hits
    ]
    assert len(hits) == 2


def test_similar_on_an_unknown_folder_returns_nothing(two_folders):
    library, _ = make_library(QUERIES)
    assert library.similar("nosuchkey", 1, k=5) == []


def test_forget_drops_a_folder_from_search(two_folders):
    library, _ = make_library(QUERIES)
    trips = next(f for f in library.folders() if f.name == "trips")

    assert library.forget(trips.key) is True
    assert [f.name for f in library.folders()] == ["walks"]
    assert not any(h.folder_name == "trips" for h in library.search("beach", k=10))
    assert (two_folders / "trips").is_dir(), "the photos stay"


def test_forget_an_unknown_folder_reports_false(two_folders):
    library, _ = make_library(QUERIES)
    assert library.forget("nosuchkey") is False


def test_empty_library_searches_cleanly(tmp_path):
    library, _ = make_library(QUERIES)
    assert library.folders() == []
    assert library.search("beach", k=10) == []
    assert library.size == 0
