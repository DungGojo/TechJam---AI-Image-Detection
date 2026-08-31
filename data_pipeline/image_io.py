"""Safe image loading and normalization."""

from __future__ import annotations

import warnings
from pathlib import Path

import PIL
from PIL import Image, ImageFile, ImageOps

# Explicit decompression-bomb ceiling.
Image.MAX_IMAGE_PIXELS = 300_000_000

# Crop oversized inputs to preserve native pixel statistics.
MAX_SIDE = 8000

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ImageLoadError(Exception):
    """Raised when a file cannot be turned into a usable RGB image."""


def resolve_path(path: str | Path) -> Path:
    """Resolve portable manifest paths against the repository root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return PROJECT_ROOT / candidate


def is_image_path(path: str | Path) -> bool:
    """Extension test, case-insensitive (SETUP.md §7.1)."""
    return Path(path).suffix.lower() in SUPPORTED_SUFFIXES


def iter_image_paths(root: str | Path) -> list[Path]:
    """Yield supported images recursively in deterministic order."""
    root = Path(root)
    if root.is_file():
        return [root] if is_image_path(root) else []
    return sorted(p for p in root.rglob("*") if p.is_file() and is_image_path(p))


def _flatten_alpha(img: Image.Image) -> Image.Image:
    """Composite any transparency onto white, then drop the alpha channel."""
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img.convert("RGBA"))
    return img


def _centre_crop_to_limit(img: Image.Image) -> Image.Image:
    """Bound the longest side by cropping. Never resamples."""
    w, h = img.size
    if max(w, h) <= MAX_SIDE:
        return img
    new_w, new_h = min(w, MAX_SIDE), min(h, MAX_SIDE)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


def _finalise(img: Image.Image) -> Image.Image:
    """EXIF rotation -> alpha flatten -> RGB -> size ceiling."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rotated = ImageOps.exif_transpose(img)
    img = rotated if rotated is not None else img
    img = _flatten_alpha(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return _centre_crop_to_limit(img)


def load_image(path: str | Path, *, salvage_truncated: bool = True) -> Image.Image:
    """Load an RGB image, retrying truncated files once."""
    path = resolve_path(path)
    try:
        with Image.open(path) as img:
            img.load()
            return _finalise(img)
    except (OSError, SyntaxError, ValueError, PIL.UnidentifiedImageError) as exc:
        if not salvage_truncated:
            raise ImageLoadError(f"{path}: {exc.__class__.__name__}: {exc}") from exc

    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(path) as img:
            img.load()
            return _finalise(img)
    except Exception as exc:  # noqa: BLE001 - last resort, report and give up
        raise ImageLoadError(f"{path}: {exc.__class__.__name__}: {exc}") from exc
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def load_image_safe(path: str | Path) -> tuple[Image.Image | None, str | None]:
    """Non-raising variant: returns (image, None) or (None, error_message)."""
    try:
        return load_image(path), None
    except ImageLoadError as exc:
        return None, str(exc)


def image_size(path: str | Path) -> tuple[int, int]:
    """Return the EXIF-aware native width and height without decoding pixels."""
    with Image.open(resolve_path(path)) as img:
        w, h = img.size
        orientation = 1
        try:
            exif = img.getexif()
            orientation = int(exif.get(0x0112, 1) or 1)
        except Exception:  # noqa: BLE001 - missing/!broken EXIF is normal
            orientation = 1
    return (h, w) if orientation in (5, 6, 7, 8) else (w, h)
