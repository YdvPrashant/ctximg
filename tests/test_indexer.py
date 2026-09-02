"""The indexing pipeline against real files on disk, with a stubbed model."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ctximg import images as imglib
from ctximg import paths
from ctximg.config import Config
from ctximg.indexer import Progress, run_index
from ctximg.store import Store


class FakeEmbedder:
    """Deterministic stand-in: a photo's vector is its average colour."""

    model_id = "fake/v1"
    dim = 4
    device = "cpu"

    def preprocess(self, image):
        return np.asarray(image.resize((4, 4)), dtype=np.float32).mean(axis=(0, 1))

    def encode_images(self, tensors):
        arr = np.stack([np.append(t, 1.0) for t in tensors]).astype(np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)

    def encode_query(self, query, ensemble=True):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def make_photo(path, colour=(200, 30, 30), size=(64, 48), exif=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, colour)
    if exif is not None:
        image.save(path, "JPEG", exif=exif)
    else:
        image.save(path, "JPEG")
    return path


@pytest.fixture
def setup(tmp_path):
    gallery = tmp_path / "gallery"
    make_photo(gallery / "red.jpg", (200, 30, 30))
    make_photo(gallery / "trip" / "blue.jpg", (30, 30, 200))

    cfg = Config()
    store = Store(paths.db_path())
    yield store, cfg, gallery, paths.thumbs_dir()
    store.close()


def index(store, cfg, thumbs, roots=None, **kw):
    return run_index(store, FakeEmbedder(), cfg, thumbs, roots or [], **kw)


def test_first_run_embeds_everything_and_writes_thumbnails(setup):
    store, cfg, gallery, thumbs = setup
    result = index(store, cfg, thumbs, [gallery])

    assert result.sync.added == 2
    assert result.embedded == 2
    assert result.failed == 0
    assert store.stats().embedded == 2

    for image_id, _ in store.get_images([1, 2]).items():
        assert imglib.thumb_path(thumbs, image_id).exists()


def test_second_run_is_a_no_op(setup):
    store, cfg, gallery, thumbs = setup
    index(store, cfg, thumbs, [gallery])
    result = index(store, cfg, thumbs, [gallery])

    assert (result.sync.added, result.sync.updated, result.embedded) == (0, 0, 0)
    assert result.sync.unchanged == 2


def test_only_the_new_photo_is_embedded_on_rerun(setup):
    store, cfg, gallery, thumbs = setup
    index(store, cfg, thumbs, [gallery])

    make_photo(gallery / "green.jpg", (30, 200, 30))
    result = index(store, cfg, thumbs, [gallery])

    assert result.sync.added == 1
    assert result.embedded == 1, "existing photos must not be re-embedded"
    assert store.stats().embedded == 3


def test_deleted_photo_leaves_the_index(setup):
    store, cfg, gallery, thumbs = setup
    index(store, cfg, thumbs, [gallery])
    (gallery / "red.jpg").unlink()

    result = index(store, cfg, thumbs, [gallery])
    assert result.sync.removed == 1
    assert store.stats().total == 1
    assert len(store.load_matrix()[0]) == 1


def test_corrupt_file_is_recorded_and_never_retried(setup):
    store, cfg, gallery, thumbs = setup
    (gallery / "broken.jpg").write_bytes(b"this is not a JPEG")

    result = index(store, cfg, thumbs, [gallery])
    assert result.failed == 1
    assert result.embedded == 2
    assert store.stats().failed == 1

    again = index(store, cfg, thumbs, [gallery])
    assert again.embedded == 0, "a known-bad file must not be retried"
    assert again.failed == 0


def test_rebuild_re_embeds_under_the_new_model(setup):
    store, cfg, gallery, thumbs = setup
    index(store, cfg, thumbs, [gallery])

    class OtherEmbedder(FakeEmbedder):
        model_id = "other/v2"

    with pytest.raises(Exception, match="--rebuild"):
        run_index(store, OtherEmbedder(), cfg, thumbs, [gallery])

    result = run_index(store, OtherEmbedder(), cfg, thumbs, [gallery], rebuild=True)
    assert result.embedded == 2
    assert store.get_meta("model_id") == "other/v2"


def test_progress_is_reported_and_reset(setup):
    store, cfg, gallery, thumbs = setup
    progress = Progress()
    index(store, cfg, thumbs, [gallery], progress=progress)

    assert progress.total == 2
    assert progress.done == 2
    assert progress.running is False
    assert progress.phase == "idle"


def test_batching_covers_every_photo(setup):
    store, cfg, gallery, thumbs = setup
    for i in range(11):
        make_photo(gallery / f"extra{i}.jpg", (i * 20 % 255, 40, 40))
    cfg.batch_size = 4

    result = index(store, cfg, thumbs, [gallery])
    assert result.embedded == 13
    assert len(store.load_matrix()[0]) == 13


def test_no_roots_indexes_nothing(tmp_path):
    store = Store(paths.db_path())
    result = run_index(store, FakeEmbedder(), Config(), paths.thumbs_dir(), [])
    assert result.embedded == 0 and result.sync.added == 0
    store.close()


# --- decoding -------------------------------------------------------------


def test_exif_orientation_is_applied(tmp_path):
    path = tmp_path / "sideways.jpg"
    image = Image.new("RGB", (100, 50), (10, 10, 10))
    exif = image.getexif()
    exif[274] = 6  # rotate for display
    exif[36867] = "2024:07:04 12:30:00"
    image.save(path, "JPEG", exif=exif.tobytes())

    upright, meta = imglib.open_upright(path)
    assert upright.size == (50, 100), "EXIF rotation must be baked in"
    assert (meta["width"], meta["height"]) == (50, 100)
    assert meta["taken_at"] == "2024:07:04 12:30:00"


def test_photo_without_exif_has_no_taken_at(tmp_path):
    path = make_photo(tmp_path / "plain.jpg")
    _, meta = imglib.open_upright(path)
    assert meta["taken_at"] is None
    assert (meta["width"], meta["height"]) == (64, 48)


def test_thumbnail_fits_the_requested_box(tmp_path):
    source = make_photo(tmp_path / "big.jpg", size=(1200, 800))
    image, _ = imglib.open_upright(source)
    dest = tmp_path / "thumbs" / "t.jpg"
    imglib.write_thumb(image, dest, 256)

    with Image.open(dest) as thumb:
        assert max(thumb.size) == 256
        assert thumb.width / thumb.height == pytest.approx(1200 / 800, abs=0.01)
    assert list(dest.parent.glob("*.tmp.jpg")) == []


def test_thumb_paths_are_sharded():
    a = imglib.thumb_path("/root", 5)
    b = imglib.thumb_path("/root", 261)  # 261 % 256 == 5
    assert a.parent == b.parent
    assert a.name == "5.jpg" and b.name == "261.jpg"


# --- a folder that cannot be read -----------------------------------------


def test_an_unreachable_folder_never_wipes_its_index(setup):
    """An unplugged drive must cost nothing.

    sync_files reconciles the index against what the scan finds, so a folder
    that cannot be read looks exactly like a folder whose every photo was
    deleted. Before this guard, indexing an unplugged drive destroyed the
    entire index for it.
    """
    from ctximg.indexer import FolderUnavailable

    store, cfg, gallery, thumbs = setup
    index(store, cfg, thumbs, [gallery])
    assert store.stats().embedded == 2
    before = len(store.load_matrix()[0])

    gone = gallery.parent / "unplugged"
    with pytest.raises(FolderUnavailable) as excinfo:
        index(store, cfg, thumbs, [gone])

    assert "left alone" in str(excinfo.value)
    assert str(gone) in str(excinfo.value)
    assert store.stats().total == 2, "nothing may be removed"
    assert len(store.load_matrix()[0]) == before, "vectors must survive"


def test_one_unreachable_root_stops_the_whole_run(setup):
    """Partial reachability is still ambiguous, so nothing is reconciled."""
    from ctximg.indexer import FolderUnavailable

    store, cfg, gallery, thumbs = setup
    index(store, cfg, thumbs, [gallery])

    with pytest.raises(FolderUnavailable):
        index(store, cfg, thumbs, [gallery, gallery.parent / "missing"])
    assert store.stats().total == 2
