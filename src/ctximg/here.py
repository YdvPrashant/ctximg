"""Folder mode: open a terminal anywhere and search the photos in that folder.

Running `ctximg` with no arguments treats the current directory as the gallery.
It reports whether the folder has been indexed before, offers to index it if
not, and then keeps a prompt open so the model is loaded once rather than once
per query.
"""

from __future__ import annotations

import os
import sys
import webbrowser
from pathlib import Path

from . import config as config_mod
from . import workspace
from .scan import iter_images
from .store import Store

BANNER = "  ctximg - describe what you are looking for. :help for commands."


def _fmt_count(n: int, noun: str) -> str:
    return f"{n:,} {noun}{'' if n == 1 else 's'}"


def _confirm(question: str, default: bool = True) -> bool:
    """Ask a yes/no question at the terminal.

    With no terminal to ask (piped or scripted) the answer is no, whatever the
    interactive default: indexing can run for hours, and a command that cannot
    ask must not start one on its own. Pass -y to mean yes up front.
    """
    if not sys.stdin.isatty():
        print(f"{question} no (not a terminal; pass -y to index without asking)")
        return False
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{question} {suffix} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not answer:
        return default
    return answer in ("y", "yes")


def run(folder: str | os.PathLike[str] | None = None, auto_yes: bool = False) -> int:
    """Entry point for `ctximg` with no subcommand."""
    root = Path(folder or Path.cwd()).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a folder: {root}", file=sys.stderr)
        return 1

    settings = config_mod.load()
    space = workspace.for_folder(root)
    seen_before = space.exists()

    print(f"\n  {root}")

    files = list(iter_images([root], settings.extensions, settings.ignore_dirs))
    if not files and not seen_before:
        print("  No photos here, or in any subfolder.\n")
        return 0

    # An index is only created once indexing is actually agreed. Opening a
    # Store here would leave an empty workspace behind for a folder the user
    # declined, and it would then show up in the app as a real folder.
    if not seen_before:
        print(f"  {_fmt_count(len(files), 'photo')} found, not indexed yet.\n")
        if not (auto_yes or _confirm(f"  Index {_fmt_count(len(files), 'photo')} now?")):
            print("\n  Nothing indexed. Run ctximg again when you want to.\n")
            return 0

    store = Store(space.db_path)
    try:
        store.set_meta(workspace.ROOT_KEY, str(root))
        store.commit()

        stats = store.stats()
        drift = store.preview_sync(files)

        if stats.embedded == 0:
            if seen_before:
                print(f"  {_fmt_count(len(files), 'photo')} found, not indexed yet.\n")
                if not (auto_yes
                        or _confirm(f"  Index {_fmt_count(len(files), 'photo')} now?")):
                    print("\n  Nothing indexed. Run ctximg again when you want to.\n")
                    return 0
        else:
            print(f"  {_fmt_count(stats.embedded, 'photo')} indexed and ready.")
            if drift.changed:
                parts = []
                if drift.added:
                    parts.append(f"{drift.added} new")
                if drift.updated:
                    parts.append(f"{drift.updated} changed")
                if drift.removed:
                    parts.append(f"{drift.removed} removed")
                print(f"  {', '.join(parts)} since the last index.\n")
                if not (auto_yes or _confirm("  Update the index now?")):
                    print()
                    drift = None
            else:
                print()
                drift = None

        return _search_session(root, space, settings, store, needs_index=drift is not None
                               or stats.embedded == 0)
    finally:
        store.close()


class Session:
    """One folder open in the terminal: its index, and the shared model."""

    def __init__(self, root: Path, space, settings, store):
        from .app import App

        self.root = root
        self.space = space
        self.store = store
        self.app = App(settings)
        self._index = None

    @property
    def embedder(self):
        return self.app.embedder

    @property
    def index(self):
        if self._index is None:
            from .search import SearchIndex

            self._index = SearchIndex(self.store, self.embedder)
        return self._index


def _search_session(root, space, settings, store, needs_index: bool) -> int:
    session = Session(root, space, settings, store)

    print("  Loading model...", flush=True)
    try:
        embedder = session.embedder
    except Exception as exc:
        print(f"\n  Could not load the model: {exc}\n", file=sys.stderr)
        return 1

    if needs_index and not _index_now(session):
        return 1

    if session.index.load() == 0:
        print("\n  Nothing searchable in this folder.\n")
        return 0

    print(f"  {embedder.model_id} on {embedder.device}\n")
    print(BANNER + "\n")
    return _prompt_loop(session)


def _index_now(session) -> bool:
    from tqdm import tqdm

    from .indexer import Progress, run_index

    bar = None

    def on_batch(done, total):
        nonlocal bar
        if bar is None:
            bar = tqdm(total=total, unit="img", desc="  indexing", leave=False)
        bar.update(done - bar.n)

    try:
        result = run_index(
            session.store, session.embedder, session.app.config,
            session.space.thumbs_dir, [session.root],
            progress=Progress(), on_batch=on_batch,
        )
    except KeyboardInterrupt:
        print("\n  Interrupted - progress is saved, run ctximg again to resume.\n")
        return False
    except Exception as exc:
        print(f"\n  Indexing failed: {exc}\n", file=sys.stderr)
        return False
    finally:
        if bar is not None:
            bar.close()

    session.index.invalidate()
    print(f"  {result.summary()}\n")
    return True


HELP = """
  Type any description to search, for example:
     someone laughing at a dinner table

  :open N     open result N in your photo viewer
  :web        open this folder in the ctximg app
  :reindex    pick up new or changed photos
  :help       this list
  :q          quit
"""


def _prompt_loop(session) -> int:
    hits: list = []
    root = session.root

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue

        if line in (":q", ":quit", ":exit"):
            return 0
        if line in (":help", ":h", "?"):
            print(HELP)
            continue
        if line == ":reindex":
            _index_now(session)
            continue
        if line.startswith(":open"):
            _open_result(hits, line, root)
            continue
        if line == ":web":
            open_in_app(session.space.key)
            continue
        if line.startswith(":"):
            print(f"  Unknown command {line!r}. :help for the list.")
            continue

        try:
            hits = session.index.search(line, k=20)
        except Exception as exc:
            print(f"  Search failed: {exc}")
            continue
        _print_hits(hits, root)

    return 0


def open_in_app(folder_key: str) -> None:
    """Open this folder in the ctximg app, starting the app if it is not up."""
    from . import desktop

    url = desktop.ensure_running(folder_key=folder_key)
    if url is None:
        print("  Could not start the app.")
        return
    print(f"  Opened {url}\n")


def _print_hits(hits, root: Path) -> None:
    if not hits:
        print("  Nothing indexed to search yet.\n")
        return
    width = max((len(_relative(h.path, root)) for h in hits), default=0)
    width = min(width, 70)
    for i, hit in enumerate(hits, 1):
        name = _relative(hit.path, root)
        if len(name) > width:
            name = "..." + name[-(width - 3):]
        print(f"  {i:2d}  {hit.score:.3f}  {name}")
    print()


def _relative(path: str, root: Path) -> str:
    """Paths shown relative to the folder you opened, which is what you know."""
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return path


def _open_result(hits, line: str, root: Path) -> None:
    parts = line.split()
    if len(parts) != 2 or not parts[1].isdigit():
        print("  Usage: :open 3")
        return
    number = int(parts[1])
    if not 1 <= number <= len(hits):
        print(f"  No result {number}. Search first, then :open a number from the list.")
        return
    target = hits[number - 1].path
    try:
        if hasattr(os, "startfile"):
            os.startfile(target)  # noqa: S606 - Windows shell open
        else:
            webbrowser.open(Path(target).as_uri())
    except OSError as exc:
        print(f"  Could not open it: {exc}")


