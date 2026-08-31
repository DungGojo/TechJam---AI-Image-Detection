"""Image degradation transforms and severity sampling."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Each transform maps shared severity [0, 5] onto its own parameter range.

SEVERITY_MIN, SEVERITY_MAX = 0.0, 5.0


def _lerp(lo: float, hi: float, t: float) -> float:
    return lo + (hi - lo) * t


def _t(severity: float) -> float:
    return float(np.clip(severity, SEVERITY_MIN, SEVERITY_MAX)) / SEVERITY_MAX



def jpeg(img: Image.Image, *, quality: int, rng=None) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), subsampling=2)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def blur(img: Image.Image, *, sigma: float, rng=None) -> Image.Image:
    if sigma <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def resize(img: Image.Image, *, scale: float, rng=None) -> Image.Image:
    """Downscale by `scale`, then back up to the original size. Both hops are lossy."""
    if scale >= 1.0:
        return img
    w, h = img.size
    dw, dh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = img.resize((dw, dh), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def noise(img: Image.Image, *, sigma: float, rng=None) -> Image.Image:
    """Additive Gaussian noise in [0,1] space, as the brief specifies."""
    if sigma <= 0:
        return img
    rng = rng if rng is not None else np.random.default_rng()
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr + rng.normal(0.0, float(sigma), arr.shape).astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0) * 255.0
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def jitter(img: Image.Image, *, amount: float, rng=None) -> Image.Image:
    """Brightness / contrast / saturation each scaled by 1 +/- amount, sampled per-call."""
    if amount <= 0:
        return img
    rng = rng if rng is not None else np.random.default_rng()
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = 1.0 + float(rng.uniform(-amount, amount))
        img = enhancer(img).enhance(factor)
    return img


def crop(img: Image.Image, *, keep: float, rng=None) -> Image.Image:
    """Centre crop retaining `keep` of each side. Output is smaller than input by design."""
    if keep >= 1.0:
        return img
    w, h = img.size
    nw, nh = max(1, int(round(w * keep))), max(1, int(round(h * keep)))
    left, top = (w - nw) // 2, (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh))



def webp(img: Image.Image, *, quality: int, rng=None) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def sharpen(img: Image.Image, *, amount: float, rng=None) -> Image.Image:
    if amount <= 0:
        return img
    return ImageEnhance.Sharpness(img).enhance(1.0 + float(amount))


def speckle(img: Image.Image, *, sigma: float, rng=None) -> Image.Image:
    """Multiplicative noise. Borrowed from the NTIRE 2026 third-place team."""
    if sigma <= 0:
        return img
    rng = rng if rng is not None else np.random.default_rng()
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr * (1.0 + rng.normal(0.0, float(sigma), arr.shape).astype(np.float32))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="RGB")


def color_cast(img: Image.Image, *, amount: float, rng=None) -> Image.Image:
    """Per-channel gain, the way a bad white balance or a filter app shifts an image."""
    if amount <= 0:
        return img
    rng = rng if rng is not None else np.random.default_rng()
    arr = np.asarray(img, dtype=np.float32)
    gains = 1.0 + rng.uniform(-amount, amount, size=3).astype(np.float32)
    arr = np.clip(arr * gains[None, None, :], 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")



@dataclass(frozen=True)
class Family:
    name: str
    fn: Callable[..., Image.Image]
    kwarg: str
    lo: float
    hi: float
    discrete: tuple
    integral: bool = False
    held_out: bool = False

    def value_at(self, severity: float) -> float:
        v = _lerp(self.lo, self.hi, _t(severity))
        return int(round(v)) if self.integral else float(v)

    def apply(self, img: Image.Image, value: float, rng=None) -> Image.Image:
        return self.fn(img, **{self.kwarg: value}, rng=rng)


FAMILIES: dict[str, Family] = {
    "jpeg":   Family("jpeg",   jpeg,   "quality", 95, 25,   (90, 70, 50, 30), integral=True),
    "blur":   Family("blur",   blur,   "sigma",   0.0, 2.5, (0.5, 1.0, 2.0)),
    "resize": Family("resize", resize, "scale",   1.0, 0.2, (0.5, 0.25)),
    "noise":  Family("noise",  noise,  "sigma",   0.0, 0.12,(0.02, 0.05, 0.10)),
    "jitter": Family("jitter", jitter, "amount",  0.0, 0.30,(0.20,)),
    "crop":   Family("crop",   crop,   "keep",    1.0, 0.65,(0.80,)),
    # Held-out transforms measure generalisation outside the reported grid.
    "webp":    Family("webp",    webp,    "quality", 95, 30,  (), integral=True, held_out=True),
    "sharpen": Family("sharpen", sharpen, "amount",  0.0, 1.5, (), held_out=True),
    "speckle": Family("speckle", speckle, "sigma",   0.0, 0.2, (), held_out=True),
    "cast":    Family("cast",    color_cast, "amount", 0.0, 0.25, (), held_out=True),
}

BRIEF_FAMILIES = [n for n, f in FAMILIES.items() if not f.held_out]
HELD_OUT_FAMILIES = [n for n, f in FAMILIES.items() if f.held_out]



@dataclass(frozen=True)
class Cell:
    family: str
    value: float

    @property
    def label(self) -> str:
        return f"{self.family}={self.value:g}"

    def apply(self, img: Image.Image, rng=None) -> Image.Image:
        return FAMILIES[self.family].apply(img, self.value, rng=rng)


def discrete_grid(families: Sequence[str] | None = None) -> list[Cell]:
    """Every (family, exact brief value) pair. This is the evaluation axis, nothing else."""
    names = list(families) if families else BRIEF_FAMILIES
    return [Cell(n, v) for n in names for v in FAMILIES[n].discrete]



@dataclass(frozen=True)
class Tier:
    name: str
    n_lo: int
    n_hi: int
    mean: float
    std: float


TIERS: dict[str, Tier] = {
    "clean":    Tier("clean",    0, 0, 0.0, 0.0),
    "mild":     Tier("mild",     1, 3, 0.0, 2.5),
    "moderate": Tier("moderate", 3, 6, 2.5, 2.0),
    "heavy":    Tier("heavy",    6, 6, 3.5, 1.0),
}


def sample_chain(
    rng: np.random.Generator,
    tier: str = "mild",
    pool: Iterable[str] | None = None,
) -> list[Cell]:
    """Sample an ordered chain of degradations for one training example."""
    t = TIERS[tier]
    if t.n_hi == 0:
        return []
    names = list(pool) if pool is not None else BRIEF_FAMILIES + HELD_OUT_FAMILIES
    k = int(rng.integers(t.n_lo, t.n_hi + 1)) if t.n_hi > t.n_lo else t.n_hi
    k = min(k, len(names))
    chosen = rng.choice(names, size=k, replace=False)
    chain = []
    for name in chosen:
        sev = float(np.clip(rng.normal(t.mean, t.std), SEVERITY_MIN, SEVERITY_MAX))
        chain.append(Cell(str(name), FAMILIES[str(name)].value_at(sev)))
    return chain


def apply_chain(img: Image.Image, chain: Sequence[Cell], rng=None) -> Image.Image:
    for cell in chain:
        img = cell.apply(img, rng=rng)
    return img



def tier_weights(epoch: int, total_epochs: int, schedule: str = "escalate") -> dict[str, float]:
    """Return curriculum tier probabilities for an epoch."""
    if schedule == "clean":
        return {"clean": 1.0, "mild": 0.0, "moderate": 0.0, "heavy": 0.0}
    if schedule == "fixed":
        return {"clean": 0.25, "mild": 0.25, "moderate": 0.25, "heavy": 0.25}
    if schedule != "escalate":
        raise ValueError(f"unknown schedule: {schedule}")
    p = epoch / max(1, total_epochs - 1)          # 0 -> 1 over training
    return {
        "clean":    max(0.10, 0.40 - 0.30 * p),
        "mild":     0.40 - 0.10 * p,
        "moderate": 0.15 + 0.20 * p,
        "heavy":    0.05 + 0.20 * p,
    }


def pick_tier(rng: np.random.Generator, weights: dict[str, float]) -> str:
    names = list(weights)
    probs = np.array([weights[n] for n in names], dtype=np.float64)
    probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))
