"""Desktop shortcut for the app.

The .lnk is created by driving WScript.Shell through PowerShell, which every
Windows install already has - no extra dependency for a one-off action.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from . import paths

ICON_NAME = "ctximg.ico"


def launcher() -> Path:
    """pythonw, so double-clicking the icon opens no console window."""
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    return pythonw if pythonw.exists() else Path(sys.executable)


def make_icon(dest: Path | None = None) -> Path:
    """Draw the app icon: a magnifier over the dark ground the UI uses."""
    from PIL import Image, ImageDraw

    dest = dest or paths.data_dir() / ICON_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)

    size = 256
    image = Image.new("RGBA", (size, size), (19, 19, 20, 255))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=48, fill=(19, 19, 20, 255))
    draw.ellipse((60, 52, 176, 168), outline=(237, 237, 239, 255), width=16)
    draw.line((162, 154, 206, 198), fill=(237, 237, 239, 255), width=20)
    image.save(dest, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return dest


def _shortcut_dir(desktop: bool) -> Path:
    if desktop:
        return Path(os.path.expanduser("~")) / "Desktop"
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs"


def install(desktop: bool = True) -> Path:
    """Create the shortcut and return its path."""
    if os.name != "nt":
        raise RuntimeError("Shortcuts are a Windows feature")

    folder = _shortcut_dir(desktop)
    folder.mkdir(parents=True, exist_ok=True)
    link = folder / "ctximg.lnk"
    icon = make_icon()

    script = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{link}');"
        "$s.TargetPath = '{target}';"
        "$s.Arguments = '-m ctximg.desktop';"
        "$s.WorkingDirectory = '{cwd}';"
        "$s.IconLocation = '{icon}';"
        "$s.Description = 'ctximg - search your photos by description';"
        "$s.Save()"
    ).format(
        link=str(link).replace("'", "''"),
        target=str(launcher()).replace("'", "''"),
        cwd=str(Path(sys.executable).parent).replace("'", "''"),
        icon=str(icon).replace("'", "''"),
    )

    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not link.exists():
        raise RuntimeError(result.stderr.strip() or "WScript.Shell did not create it")
    return link
