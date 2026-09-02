"""The index must track disk faithfully and never mix two vector spaces."""

from __future__ import annotations

import numpy as np
import pytest

from ctximg import paths
from ctximg.scan import FileEntry
from ctximg.store import ModelMismatch, Store


@pytest.fixture
def store():
    with Store(paths.db_path()) as s:
        yield s


def entry(path, size=100, mtime=1000.0):
    return FileEntry(str(path), size, mtime)


def test_new_files_are_added_and_queued(store):
    result = store.sync_files([entry("/a.jpg"), entry("/b.jpg")])
    assert (result.added, result.updated, result.removed) == (2, 0, 0)
    assert store.pending_count() == 2


def test_unchanged_files_are_not_requeued(store):
    files = [entry("/a.jpg"), entry("/b.jpg")]
    store.sync_files(files)
    store.save_vectors([(i, np.ones(4, np.float32), {}) for i, _ in store.pending()])

    result = store.sync_files(files)
    assert (result.added, result.updated, result.unchanged) == (0, 0, 2)
    assert store.pending_count() == 0


def test_changed_file_is_requeued_and_loses_its_stale_vector(store):
    store.sync_files([entry("/a.jpg", size=100, mtime=1000.0)])
    store.set_meta("embed_dim", "4")
    image_id = store.pending()[0][0]
    store.save_vectors([(image_id, np.ones(4, np.float32), {})])
    assert store.get_vector(image_id) is not None

    result = store.sync_files([entry("/a.jpg", size=250, mtime=1000.0)])
    assert result.updated == 1
    assert store.pending_count() == 1
    assert store.get_vector(image_id) is None, "stale vector must be dropped"


def test_touching_mtime_alone_requeues(store):
    store.sync_files([entry("/a.jpg", mtime=1000.0)])
    store.save_vectors([(store.pending()[0][0], np.ones(4, np.float32), {})])

    result = store.sync_files([entry("/a.jpg", mtime=2000.0)])
    assert result.updated == 1


def test_deleted_file_is_removed_with_its_vector(store):
    store.set_meta("embed_dim", "4")
    store.sync_files([entry("/a.jpg"), entry("/b.jpg")])
    for image_id, _ in store.pending():
        store.save_vectors([(image_id, np.ones(4, np.float32), {})])

    result = store.sync_files([entry("/a.jpg")])
    assert result.removed == 1

    ids, mat = store.load_matrix()
    assert len(ids) == 1, "the vector must go with the row (cascade delete)"
    assert mat.shape == (1, 4)


def test_errored_files_are_not_retried(store):
    store.sync_files([entry("/broken.jpg")])
    image_id = store.pending()[0][0]
    store.mark_error(image_id, "cannot identify image file")
    store.commit()

    assert store.pending_count() == 0
    assert store.stats().failed == 1
    assert store.failures()[0][1] == "cannot identify image file"


def test_vectors_round_trip_normalised(store):
    store.set_meta("embed_dim", "4")
    store.sync_files([entry("/a.jpg")])
    image_id = store.pending()[0][0]

    raw = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32)  # norm 5, not unit
    store.save_vectors([(image_id, raw, {"width": 800, "height": 600})])

    vec = store.get_vector(image_id)
    assert vec.shape == (4,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-3)
    assert np.allclose(vec, raw / 5.0, atol=1e-3), "float16 storage must stay accurate"

    meta = store.get_image(image_id)
    assert (meta["width"], meta["height"]) == (800, 600)


def test_load_matrix_rows_are_unit_and_ordered_by_id(store):
    store.set_meta("embed_dim", "3")
    store.sync_files([entry(f"/{i}.jpg") for i in range(5)])
    pending = store.pending()
    for n, (image_id, _) in enumerate(pending):
        store.save_vectors([(image_id, np.array([n + 1.0, 0.0, 0.0], np.float32), {})])

    ids, mat = store.load_matrix()
    assert list(ids) == sorted(ids)
    assert mat.shape == (5, 3)
    assert mat.dtype == np.float32
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-3)


def test_load_matrix_on_empty_index(store):
    ids, mat = store.load_matrix()
    assert len(ids) == 0
    assert mat.shape[0] == 0


def test_model_binding_is_recorded(store):
    store.ensure_model("ViT-B-32/laion2b_s34b_b79k", 512)
    store.ensure_model("ViT-B-32/laion2b_s34b_b79k", 512)  # idempotent
    assert store.get_meta("model_id") == "ViT-B-32/laion2b_s34b_b79k"
    assert store.embed_dim() == 512


def test_switching_model_raises_rather_than_mixing_spaces(store):
    store.ensure_model("ViT-B-32/laion2b_s34b_b79k", 512)
    with pytest.raises(ModelMismatch) as excinfo:
        store.ensure_model("ViT-L-14/laion2b_s32b_b82k", 768)
    assert "--rebuild" in str(excinfo.value)


def test_rebind_clears_vectors_and_requeues_everything(store):
    store.ensure_model("ViT-B-32/laion2b_s34b_b79k", 512)
    store.sync_files([entry("/a.jpg")])
    store.save_vectors([(store.pending()[0][0], np.ones(512, np.float32), {})])
    assert store.pending_count() == 0

    store.rebind_model("ViT-L-14/laion2b_s32b_b82k", 768)
    assert store.embed_dim() == 768
    assert store.pending_count() == 1
    assert len(store.load_matrix()[0]) == 0
    store.ensure_model("ViT-L-14/laion2b_s32b_b82k", 768)  # now accepted


def test_stats_counts(store):
    store.ensure_model("m", 4)
    store.sync_files([entry(f"/{i}.jpg") for i in range(4)])
    pending = store.pending()
    store.save_vectors([(pending[0][0], np.ones(4, np.float32), {})])
    store.mark_error(pending[1][0], "boom")
    store.commit()

    stats = store.stats()
    assert (stats.total, stats.embedded, stats.failed, stats.pending) == (4, 1, 1, 2)
    assert stats.model == "m" and stats.dim == 4
    assert stats.db_bytes > 0


def test_index_survives_reopening(store):
    store.ensure_model("m", 4)
    store.sync_files([entry("/a.jpg")])
    store.save_vectors([(store.pending()[0][0], np.ones(4, np.float32), {})])
    store.close()

    with Store(paths.db_path()) as reopened:
        assert reopened.stats().embedded == 1
        assert reopened.get_meta("model_id") == "m"
