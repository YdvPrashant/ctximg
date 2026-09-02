"""Command line, scoped to one folder.

The app is where you search across folders and look at results; the terminal
works on the folder you are standing in. Reading fifty filenames in a console
is miserable, which is exactly what the browser is for.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import config as config_mod
from . import paths


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _cuda_wheel_hint() -> str:
    return (
        "  .venv\\Scripts\\python -m pip uninstall -y torch torchvision\n"
        "  .venv\\Scripts\\python -m pip install torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu126"
    )


def _gpu_missed_warning(device: str) -> str | None:
    """Warn when a GPU is installed but the torch build cannot reach it.

    Auto-detection is honest about what torch reports, and a CPU-only wheel
    reports no GPU - so without this the fallback is silent and the run just
    takes ten times longer for no visible reason.
    """
    if device != "cpu" or not shutil.which("nvidia-smi"):
        return None
    return (
        "warning: an NVIDIA GPU is present but this torch build is CPU-only, "
        "so indexing will run far slower than it could.\n"
        "  Install the CUDA build, then re-run with --rebuild:\n" + _cuda_wheel_hint()
    )


def _resolve(folder) -> Path | None:
    root = Path(folder or Path.cwd()).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a folder: {root}", file=sys.stderr)
        return None
    return root


# --- folder commands ------------------------------------------------------


def cmd_here(args) -> int:
    from . import here

    return here.run(args.folder, auto_yes=args.yes)


def cmd_index(args) -> int:
    """Index or reindex the folder you are standing in."""
    from tqdm import tqdm

    from . import workspace
    from .app import App
    from .indexer import Progress, run_index
    from .store import Store

    root = _resolve(args.folder)
    if root is None:
        return 1

    space = workspace.for_folder(root)
    app = App()

    print(f"\n  {root}")
    print("  Loading model (first run downloads it)...", flush=True)
    embedder = app.embedder
    print(f"  {embedder.model_id} on {embedder.device}")
    if (warning := _gpu_missed_warning(embedder.device)) is not None:
        print(warning, file=sys.stderr)

    bar = None

    def on_batch(done, total):
        nonlocal bar
        if bar is None:
            bar = tqdm(total=total, unit="img", desc="  indexing")
        bar.update(done - bar.n)

    store = Store(space.db_path)
    try:
        store.set_meta(workspace.ROOT_KEY, str(root))
        store.commit()
        result = run_index(
            store, embedder, app.config, space.thumbs_dir, [root],
            rebuild=args.rebuild, progress=Progress(), on_batch=on_batch,
        )
    except KeyboardInterrupt:
        print("\n  interrupted - progress is saved, re-run to resume")
        return 130
    finally:
        if bar is not None:
            bar.close()
        store.close()

    print(f"  {result.summary()}")
    if result.failed:
        print(f"  ({result.failed} unreadable file(s); see 'ctximg stats')")
    return 0


def cmd_forget(args) -> int:
    """Delete this folder's index. The photos are never touched."""
    from . import here, workspace

    root = _resolve(args.folder)
    if root is None:
        return 1

    space = workspace.for_folder(root)
    info = workspace.find(space.key)
    if info is None or not space.exists():
        print(f"\n  {root}\n  Not indexed, so there is nothing to forget.\n")
        return 0

    print(f"\n  {root}")
    print(f"  {info.photos:,} photos indexed. Your photos will not be touched.\n")
    if not (args.yes or here._confirm("  Delete this folder's index?", default=False)):
        print("\n  Kept.\n")
        return 0

    workspace.forget(info)
    print("\n  Index deleted.\n")
    return 0


def cmd_web(args) -> int:
    """Open this folder in the app."""
    from . import desktop, workspace

    root = _resolve(args.folder)
    if root is None:
        return 1

    space = workspace.for_folder(root)
    if not space.exists():
        print(f"\n  {root} is not indexed yet.")
        print("  Run 'ctximg' here first, or add it from the app.\n", file=sys.stderr)
        return 1

    url = desktop.ensure_running(folder_key=space.key)
    if url is None:
        print("Could not start the app.", file=sys.stderr)
        return 1
    print(f"\n  {url}\n")
    return 0


def cmd_app(args) -> int:
    """Launch the app itself."""
    from . import desktop

    if desktop._already_running():
        print(f"\n  Already running: {desktop.base_url()}\n")
        import webbrowser

        webbrowser.open(desktop.app_url())
        return 0
    print(f"\n  {desktop.base_url()}\n  Ctrl+C to stop.\n")
    return desktop.main()


def cmd_list(args) -> int:
    from . import workspace

    folders = workspace.known()
    if not folders:
        print("No folders indexed yet.")
        print("  cd into a folder of photos and run: ctximg")
        return 0

    print(f"\n{len(folders)} folder(s) indexed:\n")
    for info in folders:
        mark = "" if info.on_disk else "   (folder is gone)"
        print(f"  {info.photos:>7,}  {info.root}{mark}")
    print()
    return 0


# --- settings and diagnostics ---------------------------------------------


def cmd_config(args) -> int:
    cfg = config_mod.load()

    if args.config_action == "show":
        print(f"config file: {paths.config_path()}")
        print(f"indexes:     {paths.data_dir() / 'folders'}")
        print()
        for key in ("tier", "model", "device", "precision", "thumb_size", "batch_size"):
            value = getattr(cfg, key)
            print(f"  {key:12s} {value if value is not None else 'auto'}")
        print(f"  {'extensions':12s} {' '.join(cfg.extensions)}")

        from . import sysinfo
        from .embedder import describe_tiers, resolve_precision

        machine = sysinfo.machine()
        device = cfg.device if cfg.device != "auto" else ("cuda" if machine.gpu else "cpu")
        print(f"\nwould run on   {device}, {resolve_precision(cfg.precision, device)}")
        if machine.gpu:
            gpu = machine.gpu
            busy = f", {gpu.util}% busy" if gpu.util is not None else ""
            print(f"  gpu          {gpu.name}")
            print(f"               {gpu.mem_used:.1f}/{gpu.mem_total:.1f} GB used{busy}")
        print(f"  cpu          {machine.cpu_count} cores, {machine.cpu_percent:.0f}% busy")
        print(f"  memory       {machine.ram_used:.1f}/{machine.ram_total:.1f} GB")

        print("\nquality tiers")
        for tier in describe_tiers(device, cfg.rates):
            mark = "*" if tier["key"] == cfg.tier else " "
            source = "measured here" if tier["measured"] else "estimate"
            print(f" {mark} {tier['key']:9s} {tier['rate']:>6.1f} img/s ({source})")
            print(f"   {'':9s} {tier['model']}")
        print("\nChange with: ctximg config set tier <fast|balanced|best>")
        print("Folders are managed in the app, or with 'ctximg' inside a folder.")
        return 0

    value = cfg.set_value(args.key, args.value)
    config_mod.save(cfg)
    print(f"{args.key} = {value if value is not None else 'auto'}")
    if args.key in ("model", "device"):
        print("Changing this invalidates existing vectors: reindex your folders "
              "(the app offers this per folder).")
    return 0


def cmd_stats(args) -> int:
    from . import workspace
    from .store import Store

    root = _resolve(args.folder)
    if root is None:
        return 1
    space = workspace.for_folder(root)
    if not space.exists():
        print(f"\n  {root}\n  Not indexed yet.\n")
        return 0

    with Store(space.db_path) as store:
        stats = store.stats()
        print(f"\n  {root}")
        print(f"  photos found     {stats.total}")
        print(f"    embedded       {stats.embedded}")
        print(f"    pending        {stats.pending}")
        print(f"    unreadable     {stats.failed}")
        print(f"  model            {stats.model or '(none yet)'}")
        print(f"  index size       {_fmt_bytes(stats.db_bytes)}")
        failures = store.failures(10)
        if failures:
            print("\n  unreadable files:")
            for path, error in failures:
                print(f"    {path}\n      {error}")
    print()
    return 0


def cmd_doctor(args) -> int:
    cfg = config_mod.load()
    print(f"python           {sys.version.split()[0]} ({sys.executable})")

    try:
        import torch
    except ImportError:
        print("torch            NOT INSTALLED - run: pip install -e .")
        return 1

    # A failed uninstall on Windows can leave an empty torch\ directory behind.
    # Python imports that as a namespace package, so `import torch` succeeds
    # and every attribute access then fails - report it instead of crashing.
    if not hasattr(torch, "version"):
        location = (list(getattr(torch, "__path__", [])) or ["site-packages"])[0]
        print("torch            BROKEN INSTALL")
        print(
            f"\nAn empty torch directory is shadowing the real package:\n"
            f"  {location}\n\n"
            "It is left over from an interrupted uninstall. Delete it, then "
            "reinstall:\n"
            f'  Remove-Item -Recurse -Force "{location}"\n' + _cuda_wheel_hint()
        )
        return 1

    available = torch.cuda.is_available()
    print(f"torch            {torch.__version__}")
    print(f"cuda build       {torch.version.cuda or 'cpu-only wheel'}")
    print(f"cuda available   {available}")
    if available:
        print(f"gpu              {torch.cuda.get_device_name(0)}")

    device = "cuda" if available and cfg.device != "cpu" else "cpu"
    try:
        from .embedder import resolve_model

        print(f"device           {device}")
        print(f"model            {resolve_model(cfg.model, device).model_id}")
    except Exception as exc:
        print(f"model            ERROR: {exc}")

    if not available and shutil.which("nvidia-smi"):
        print(
            "\nAn NVIDIA driver is present but this torch build is CPU-only, so "
            "indexing will run roughly 10x slower than it could.\n"
            "  Reinstall with CUDA:\n" + _cuda_wheel_hint()
        )

    from . import workspace

    folders = workspace.known()
    print(f"\nindexed folders  {len(folders)}")
    for info in folders[:10]:
        mark = "" if info.on_disk else "  (gone)"
        print(f"  {info.photos:>7,}  {info.root}{mark}")
    return 0


def cmd_install_shortcut(args) -> int:
    from . import shortcut

    try:
        link = shortcut.install(desktop=not args.start_menu)
    except Exception as exc:
        print(f"Could not create the shortcut: {exc}", file=sys.stderr)
        return 1
    print(f"\n  Shortcut created: {link}")
    print("  Double-click it to open ctximg.\n")
    return 0



def cmd_authorship(args) -> int:
    """Show, record or check the authorship proof."""
    import getpass

    from . import authorship

    if args.authorship_action in (None, "show"):
        print(f"\n  {authorship.PROJECT} was written by {authorship.AUTHOR}.")
        if authorship.is_set():
            print(f"  Authorship proof recorded in {authorship.source_file()}")
            print("  Check it with: ctximg authorship verify\n")
        else:
            print("  No authorship proof recorded yet.")
            print("  Record one with: ctximg authorship set\n")
        return 0

    if not sys.stdin.isatty():
        print("This needs a terminal so the secret is never echoed or piped.",
              file=sys.stderr)
        return 1

    if args.authorship_action == "set":
        if authorship.is_set() and not args.force:
            print("\n  A proof is already recorded. Replace it with --force.\n",
                  file=sys.stderr)
            return 1
        secret = getpass.getpass("  Secret (not shown): ")
        if not secret:
            print("  Nothing entered.", file=sys.stderr)
            return 1
        if secret != getpass.getpass("  Again to confirm: "):
            print("  Those did not match.", file=sys.stderr)
            return 1
        where = authorship.record(authorship.make_commitment(secret))
        print(f"\n  Recorded in {where}")
        print("  Keep the secret somewhere safe: it is the only way to prove")
        print("  this later, and it cannot be recovered from the file.\n")
        return 0

    if args.authorship_action == "verify":
        if not authorship.is_set():
            print("\n  No proof recorded yet. Run: ctximg authorship set\n",
                  file=sys.stderr)
            return 1
        if authorship.verify(getpass.getpass("  Secret (not shown): ")):
            print(f"\n  Verified: this is {authorship.AUTHOR}'s work.\n")
            return 0
        print("\n  That secret does not match the recorded proof.\n", file=sys.stderr)
        return 1

    return 1


# --- parser ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ctximg",
        description=(
            "Contextual image search. Run with no arguments inside a folder of "
            "photos to search it, or 'ctximg app' for the full app."
        ),
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="answer yes to prompts",
    )
    subs = parser.add_subparsers(dest="command", required=False)

    def folder_arg(p):
        p.add_argument("folder", nargs="?", default=None,
                       help="folder to act on (default: the current one)")
        p.add_argument("-y", "--yes", action="store_true", dest="yes",
                       default=argparse.SUPPRESS, help="answer yes to prompts")
        return p

    folder_arg(subs.add_parser("here", help="search this folder in the terminal")
               ).set_defaults(func=cmd_here)

    index_parser = folder_arg(subs.add_parser("index", help="index or reindex this folder"))
    index_parser.add_argument("--rebuild", action="store_true",
                              help="discard every vector and re-embed")
    index_parser.set_defaults(func=cmd_index)

    folder_arg(subs.add_parser("forget", help="delete this folder's index")
               ).set_defaults(func=cmd_forget)
    folder_arg(subs.add_parser("web", help="open this folder in the app")
               ).set_defaults(func=cmd_web)
    folder_arg(subs.add_parser("stats", help="index statistics for this folder")
               ).set_defaults(func=cmd_stats)

    subs.add_parser("app", help="launch the ctximg app").set_defaults(func=cmd_app)
    subs.add_parser("list", help="folders indexed so far").set_defaults(func=cmd_list)

    config_parser = subs.add_parser("config", help="view and change settings")
    config_subs = config_parser.add_subparsers(dest="config_action", required=True)
    config_subs.add_parser("show", help="print the current settings")
    set_parser = config_subs.add_parser("set", help="set an option")
    set_parser.add_argument("key")
    set_parser.add_argument("value")
    config_parser.set_defaults(func=cmd_config)

    author = subs.add_parser(
        "authorship", help="who wrote this, and proof of it"
    )
    author_subs = author.add_subparsers(dest="authorship_action", required=False)
    author_subs.add_parser("show", help="print the author")
    setter = author_subs.add_parser("set", help="record an authorship proof")
    setter.add_argument("--force", action="store_true",
                        help="replace an existing proof")
    author_subs.add_parser("verify", help="check a secret against the proof")
    author.set_defaults(func=cmd_authorship, force=False)

    subs.add_parser("doctor", help="check the environment").set_defaults(func=cmd_doctor)

    link = subs.add_parser("install-shortcut", help="put a ctximg icon on your desktop")
    link.add_argument("--start-menu", action="store_true",
                      help="install to the Start menu instead of the desktop")
    link.set_defaults(func=cmd_install_shortcut)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if getattr(args, "func", None) is None:
            from . import here

            return here.run(None, auto_yes=args.yes)
        return args.func(args)
    except config_mod.ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        from .embedder import ModelError
        from .indexer import FolderUnavailable
        from .store import ModelMismatch

        if isinstance(exc, (ModelError, ModelMismatch, FolderUnavailable)):
            print(f"error: {exc}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
