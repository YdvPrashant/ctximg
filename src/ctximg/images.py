"""Decoding photos off disk, and the thumbnail cache.

One decode serves both purposes: the tensor handed to the model and the
thumbnail written for the web UI. Decoding is the expensive part of indexing,
so doing it twice would roughly double the cost.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

try:  # iPhone photos
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the wheel being installed
    HEIF_AVAILABLE = False

# Panoramas and scans legitimately exceed Pillow's default bomb threshold.
Image.MAX_IMAGE_PIXELS = 500_000_000

EXIF_DATETIME_ORIGINAL = 36867
EXIF_DATETIME = 306


def thumb_path(root: Path, image_id: int) -> Path:
    """Sharded so a 100k-photo gallery never puts 100k files in one directory."""
    return Path(root) / f"{image_id % 256:02x}" / f"{image_id}.jpg"


def read_taken_at(image: Image.Image) -> str | None:
    """EXIF capture time, for display only - it never affects ranking."""
    try:
        exif = image.getexif()
    except Exception:
        return None
    if not exif:
        return None
    for tag in (EXIF_DATETIME_ORIGINAL, EXIF_DATETIME):
        value = exif.get(tag)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def open_upright(path: str | Path) -> tuple[Image.Image, dict]:
    """Open an image, honour its EXIF orientation, and return it with metadata.

    Phone photos carry rotation in EXIF rather than in the pixels. Ignoring it
    would feed the model sideways images and produce sideways thumbnails.
    """
    with Image.open(path) as raw:
        raw.load()
        meta = {
            "width": raw.width,
            "height": raw.height,
            "taken_at": read_taken_at(raw),
        }
        image = ImageOps.exif_transpose(raw) or raw
        image = image.convert("RGB")
    meta["width"], meta["height"] = image.size
    return image, meta


def write_thumb(image: Image.Image, dest: Path, size: int = 256) -> None:
    """Write a cached thumbnail. Failures here must not fail the indexing run."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    thumb = image.copy()
    thumb.thumbnail((size, size), Image.LANCZOS)
    tmp = dest.with_suffix(".tmp.jpg")
    thumb.save(tmp, "JPEG", quality=82, optimize=True)
    tmp.replace(dest)
