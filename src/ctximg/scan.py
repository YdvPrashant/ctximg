"""Recursive gallery walk.

Yields one entry per candidate image file. Unreadable directories are skipped
rather than aborting the scan - a gallery on an external drive or with a
locked system folder should still index everything else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class FileEntry:
    path: str      # absolute, as reported by the OS
    size: int
    mtime: float


def iter_images(
    roots: Iterable[str | os.PathLike[str]],
    extensions: Iterable[str],
    ignore_dirs: Iterable[str] = (),
) -> Iterator[FileEntry]:
    """Walk roots depth-first, yielding image files exactly once each."""
    exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in extensions}
    ignored = {d.lower() for d in ignore_dirs}
    seen: set[str] = set()

    for root in roots:
        start = Path(root)
        if not start.is_dir():
            continue
        stack = [str(start.resolve())]
        while stack:
            current = stack.pop()
            try:
                entries = list(os.scandir(current))
            except (PermissionError, OSError):
                continue
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        name = entry.name
                        if name.lower() in ignored or name.startswith("."):
                            continue
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        if os.path.splitext(entry.name)[1].lower() not in exts:
                            continue
                        if entry.path in seen:
                            continue
                        stat = entry.stat()
                        seen.add(entry.path)
                        yield FileEntry(entry.path, stat.st_size, stat.st_mtime)
                except OSError:
                    continue
