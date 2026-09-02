"""Per-folder workspaces and the read-only drift check behind folder mode."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ctximg import here, paths, workspace
from ctximg.scan import FileEntry
from ctximg.store import Store


# --- keys -----------------------------------------------------------------


def test_same_folder_always_resolves_to_the_same_index(tmp_path):
    gallery = tmp_path / "photos"
    gallery.mkdir()
    assert workspace.for_folder(gallery).key == workspace.for_folder(gallery).key


def test_a_relative_path_and_its_absolute_form_agree(tmp_path, monkeypatch):
    gallery = tmp_path / "photos"
    gallery.mkdir()
    monkeypatch.chdir(tmp_path)
    assert workspace.for_folder("photos").key == workspace.for_folder(gallery).key


def test_different_folders_get_different_indexes(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, b = workspace.for_folder(tmp_path / "a"), workspace.for_folder(tmp_path / "b")
    assert a.key != b.key
    assert a.db_path != b.db_path


@pytest.mark.skipif(os.name != "nt", reason="Windows paths are case-insensitive")
def test_case_differences_are_the_same_folder_on_windows(tmp_path):
    gallery = tmp_path / "Photos"
    gallery.mkdir()
    upper = workspace.for_folder(str(gallery).upper())
    lower = workspace.for_folder(str(gallery).lower())
    assert upper.key == lower.key


def test_workspace_dir_is_readable_and_scoped_to_the_data_dir(tmp_path):
    gallery = tmp_path / "College Photos!"
    gallery.mkdir()
    space = workspace.for_folder(gallery)
    assert "college-photos" in space.dir.name
    assert paths.data_dir() in space.dir.parents
    assert space.thumbs_dir.parent == space.dir


def test_nothing_is_created_just_by_resolving(tmp_path):
    gallery = tmp_path / "photos"
    gallery.mkdir()
    space = workspace.for_folder(gallery)
    assert not space.exists()
    assert not space.dir.exists()
    assert list(gallery.iterdir()) == [], "the photo folder must stay untouched"


def test_exists_becomes_true_once_indexed(tmp_path):
    gallery = tmp_path / "photos"
    gallery.mkdir()
    space = workspace.for_folder(gallery)
    Store(space.db_path).close()
    assert space.exists()


# --- the registry ---------------------------------------------------------


def seed(gallery, model="stub/v1", dim=3, photos=1):
    """Create a workspace with some vectors in it."""
    space = workspace.for_folder(gallery)
    with Store(space.db_path) as store:
        store.set_meta(workspace.ROOT_KEY, str(Path(gallery).resolve()))
        store.ensure_model(model, dim)
        store.sync_files([FileEntry(str(Path(gallery) / f"{i}.jpg"), 10, 1.0)
                          for i in range(photos)])
        for image_id, _ in store.pending():
            store.save_vectors([(image_id, np.ones(dim, np.float32), {})])
    return space


def test_known_lists_indexed_folders_and_ignores_the_rest(tmp_path):
    assert workspace.known() == []

    for name in ("alpha", "beta"):
        gallery = tmp_path / name
        gallery.mkdir()
        seed(gallery)

    (paths.data_dir() / "folders" / "junk").mkdir(parents=True, exist_ok=True)

    assert sorted(i.root.name for i in workspace.known()) == ["alpha", "beta"]


def test_known_skips_a_workspace_with_no_vectors(tmp_path):
    """An index created and then abandoned is not a folder the user indexed."""
    gallery = tmp_path / "abandoned"
    gallery.mkdir()
    space = workspace.for_folder(gallery)
    with Store(space.db_path) as store:
        store.set_meta(workspace.ROOT_KEY, str(gallery.resolve()))
        store.commit()

    assert workspace.known() == []
    assert len(workspace.known(include_empty=True)) == 1


def test_known_reports_model_and_photo_count(tmp_path):
    gallery = tmp_path / "counted"
    gallery.mkdir()
    seed(gallery, model="stub/v9", dim=3, photos=4)

    info = workspace.known()[0]
    assert info.photos == 4
    assert info.model == "stub/v9"
    assert info.dim == 3
    assert info.on_disk is True


def test_known_survives_the_folder_being_deleted(tmp_path):
    gallery = tmp_path / "gone"
    gallery.mkdir()
    seed(gallery)
    for child in gallery.iterdir():
        child.unlink()
    gallery.rmdir()

    found = workspace.known()
    assert len(found) == 1
    assert found[0].on_disk is False, "reported so the caller can flag it as missing"


def test_find_by_key(tmp_path):
    gallery = tmp_path / "findme"
    gallery.mkdir()
    space = seed(gallery)
    assert workspace.find(space.key).root == gallery.resolve()
    assert workspace.find("nosuchkey") is None


def test_forget_deletes_the_index_and_leaves_the_photos(tmp_path):
    gallery = tmp_path / "keepphotos"
    gallery.mkdir()
    (gallery / "photo.jpg").write_bytes(b"pretend jpeg")
    space = seed(gallery)

    info = workspace.find(space.key)
    assert workspace.forget(info) is True
    assert not space.dir.exists()
    assert (gallery / "photo.jpg").exists(), "photos must never be touched"
    assert workspace.known() == []


def test_forget_refuses_to_delete_outside_the_index_store(tmp_path):
    """A corrupted record must not be able to point rmtree at a photo folder."""
    victim = tmp_path / "precious"
    victim.mkdir()
    (victim / "photo.jpg").write_bytes(b"pretend jpeg")

    rogue = workspace.FolderInfo(
        key="x", root=victim, dir=victim, model=None, dim=None, photos=0
    )
    with pytest.raises(ValueError, match="refusing to delete"):
        workspace.forget(rogue)
    assert (victim / "photo.jpg").exists()


# --- read-only drift check ------------------------------------------------


@pytest.fixture
def store(tmp_path):
    with Store(paths.db_path()) as s:
        s.ensure_model("stub/v1", 3)
        s.sync_files([FileEntry("/a.jpg", 10, 1.0), FileEntry("/b.jpg", 10, 1.0)])
        for image_id, _ in s.pending():
            s.save_vectors([(image_id, np.ones(3, np.float32), {})])
        yield s


def test_preview_reports_no_drift_when_nothing_changed(store):
    same = [FileEntry("/a.jpg", 10, 1.0), FileEntry("/b.jpg", 10, 1.0)]
    preview = store.preview_sync(same)
    assert preview.changed == 0
    assert preview.unchanged == 2


def test_preview_sees_new_changed_and_removed(store):
    preview = store.preview_sync(
        [FileEntry("/a.jpg", 999, 1.0), FileEntry("/c.jpg", 10, 1.0)]
    )
    assert (preview.added, preview.updated, preview.removed) == (1, 1, 1)


def test_preview_changes_nothing(store):
    before = store.stats()
    version = store.vectors_version()

    store.preview_sync([FileEntry("/c.jpg", 10, 1.0)])

    after = store.stats()
    assert (after.total, after.embedded) == (before.total, before.embedded)
    assert store.vectors_version() == version, "looking must not bump the index"
    assert len(store.load_matrix()[0]) == 2


# --- prompt helpers -------------------------------------------------------


def test_paths_are_shown_relative_to_the_folder_you_opened():
    root = Path("/gallery")
    assert here._relative(str(root / "trip" / "a.jpg"), root) == str(
        Path("trip") / "a.jpg"
    )


def test_a_path_outside_the_folder_is_shown_in_full():
    assert here._relative("/elsewhere/a.jpg", Path("/gallery")) == "/elsewhere/a.jpg"


def test_counts_read_naturally():
    assert here._fmt_count(1, "photo") == "1 photo"
    assert here._fmt_count(1500, "photo") == "1,500 photos"


def test_confirm_says_no_when_there_is_no_terminal_to_ask(monkeypatch):
    """A piped or scripted run must never start a long index on its own."""
    monkeypatch.setattr(here.sys.stdin, "isatty", lambda: False)
    assert here._confirm("index?", default=True) is False
    assert here._confirm("index?", default=False) is False


def test_piped_run_reports_but_does_not_index(tmp_path, monkeypatch, capsys):
    from PIL import Image

    gallery = tmp_path / "photos"
    gallery.mkdir()
    Image.new("RGB", (32, 32), (200, 30, 30)).save(gallery / "a.jpg", "JPEG")
    monkeypatch.setattr(here.sys.stdin, "isatty", lambda: False)

    assert here.run(gallery) == 0
    out = capsys.readouterr().out
    assert "1 photo found, not indexed yet" in out
    assert "pass -y" in out

    space = workspace.for_folder(gallery)
    with Store(space.db_path) as store:
        assert store.stats().embedded == 0, "nothing may be embedded without consent"


def test_folder_mode_reports_an_empty_folder_without_indexing(tmp_path, capsys):
    empty = tmp_path / "no_photos"
    empty.mkdir()

    assert here.run(empty) == 0
    assert "No photos here" in capsys.readouterr().out
    assert not workspace.for_folder(empty).exists()


def test_folder_mode_rejects_a_path_that_is_not_a_folder(tmp_path):
    afile = tmp_path / "x.txt"
    afile.write_text("hi")
    assert here.run(afile) == 1
