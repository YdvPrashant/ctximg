"""The walk must find every photo, recurse, and quietly skip the rest."""

from __future__ import annotations

from ctximg import config
from ctximg.scan import iter_images


def names(entries):
    import os
    return sorted(os.path.basename(e.path) for e in entries)


def test_finds_images_recursively_and_ignores_other_files(gallery):
    found = list(iter_images([gallery], config.IMAGE_EXTENSIONS, config.IGNORE_DIRS))
    assert names(found) == ["a.jpg", "b.PNG", "c.jpeg", "d.heic"]


def test_extension_match_is_case_insensitive(gallery):
    found = names(iter_images([gallery], [".png"], config.IGNORE_DIRS))
    assert found == ["b.PNG"]


def test_extensions_without_leading_dot_still_work(gallery):
    found = names(iter_images([gallery], ["jpg"], config.IGNORE_DIRS))
    assert found == ["a.jpg"]


def test_hidden_and_ignored_directories_are_skipped(gallery):
    found = names(iter_images([gallery], config.IMAGE_EXTENSIONS, config.IGNORE_DIRS))
    assert "secret.jpg" not in found, "dot-directories must be skipped"
    assert "vendor.jpg" not in found, "ignore_dirs must be skipped"


def test_ignore_list_is_case_insensitive(gallery):
    found = names(iter_images([gallery], config.IMAGE_EXTENSIONS, ["NODE_MODULES"]))
    assert "vendor.jpg" not in found


def test_entries_carry_size_and_mtime(gallery):
    entry = next(e for e in iter_images([gallery], [".jpg"], config.IGNORE_DIRS))
    import os
    stat = os.stat(entry.path)
    assert entry.size == stat.st_size
    assert entry.mtime == stat.st_mtime


def test_missing_root_is_skipped_not_fatal(gallery, tmp_path):
    found = names(iter_images([tmp_path / "gone", gallery], [".jpg"], config.IGNORE_DIRS))
    assert found == ["a.jpg"]


def test_overlapping_roots_yield_each_file_once(gallery):
    roots = [gallery, gallery / "trip"]
    found = names(iter_images(roots, config.IMAGE_EXTENSIONS, config.IGNORE_DIRS))
    assert found == ["a.jpg", "b.PNG", "c.jpeg", "d.heic"]


def test_empty_root_list_yields_nothing():
    assert list(iter_images([], config.IMAGE_EXTENSIONS)) == []
