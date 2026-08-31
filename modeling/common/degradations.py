"""Image degradation transforms and severity sampling."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Each transform maps shared severity [0, 5] onto its own parameter range.

SEVERITY_MIN, SEVERITY_MAX = 0.0, 5.0


def _linear_interpolate(low_val: float, high_val: float, weight: float) -> float:
    """Linear interpolation between low_val and high_val by weight in [0, 1]."""
    return low_val + (high_val - low_val) * weight


def _normalize_severity(severity: float) -> float:
    """Normalize severity score from [SEVERITY_MIN, SEVERITY_MAX] to [0.0, 1.0]."""
    return float(np.clip(severity, SEVERITY_MIN, SEVERITY_MAX)) / SEVERITY_MAX


def jpeg(
    img: Image.Image, *, quality: int, rng: np.random.Generator | None = None
) -> Image.Image:
    """Apply lossy JPEG compression at the given quality level (1-100)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), subsampling=2)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def blur(
    img: Image.Image, *, sigma: float, rng: np.random.Generator | None = None
) -> Image.Image:
    """Apply Gaussian blur with radius sigma."""
    if sigma <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def resize(
    img: Image.Image, *, scale: float, rng: np.random.Generator | None = None
) -> Image.Image:
    """Downscale by `scale`, then back up to the original size. Both hops are lossy."""
    if scale >= 1.0:
        return img
    width, height = img.size
    downscaled_w = max(1, int(round(width * scale)))
    downscaled_h = max(1, int(round(height * scale)))
    small = img.resize((downscaled_w, downscaled_h), Image.BICUBIC)
    return small.resize((width, height), Image.BICUBIC)


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
    for enhancer in (
        ImageEnhance.Brightness,
        ImageEnhance.Contrast,
        ImageEnhance.Color,
    ):
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


@dataclass(frozen=True)
class Family:
    """A degradation transform family (e.g., jpeg, blur, resize) mapped to severity."""

    name: str
    fn: Callable[..., Image.Image]
    kwarg: str
    lo: float
    hi: float
    discrete: tuple
    integral: bool = False

    def value_at(self, severity: float) -> float:
        """Map continuous severity in [0, 5] to this family's specific parameter range."""
        interpolated = _linear_interpolate(
            self.lo, self.hi, _normalize_severity(severity)
        )
        return int(round(interpolated)) if self.integral else float(interpolated)

    def apply(
        self, img: Image.Image, value: float, rng: np.random.Generator | None = None
    ) -> Image.Image:
        """Apply the degradation transform with parameter value."""
        return self.fn(img, **{self.kwarg: value}, rng=rng)


FAMILIES: dict[str, Family] = {
    "jpeg": Family("jpeg", jpeg, "quality", 95, 25, (90, 70, 50, 30), integral=True),
    "blur": Family("blur", blur, "sigma", 0.0, 2.5, (0.5, 1.0, 2.0)),
    "resize": Family("resize", resize, "scale", 1.0, 0.2, (0.5, 0.25)),
    "noise": Family("noise", noise, "sigma", 0.0, 0.12, (0.02, 0.05, 0.10)),
    "jitter": Family("jitter", jitter, "amount", 0.0, 0.30, (0.20,)),
    "crop": Family("crop", crop, "keep", 1.0, 0.65, (0.80,)),
}

BRIEF_FAMILIES: list[str] = list(FAMILIES.keys())



@dataclass(frozen=True)
class Cell:
    """An exact degradation family and parameter value."""

    family: str
    value: float

    @property
    def label(self) -> str:
        return f"{self.family}={self.value:g}"

    def apply(
        self, img: Image.Image, rng: np.random.Generator | None = None
    ) -> Image.Image:
        return FAMILIES[self.family].apply(img, self.value, rng=rng)


def discrete_grid(families: Sequence[str] | None = None) -> list[Cell]:
    """Every (family, exact brief value) pair. This is the evaluation axis, nothing else."""
    names = list(families) if families else BRIEF_FAMILIES
    return [Cell(name, val) for name in names for val in FAMILIES[name].discrete]


@dataclass(frozen=True)
class Tier:
    """A curriculum difficulty tier specifying chain length and severity Gaussian parameters."""

    name: str
    n_lo: int
    n_hi: int
    mean: float
    std: float


TIERS: dict[str, Tier] = {
    "clean": Tier("clean", 0, 0, 0.0, 0.0),
    "mild": Tier("mild", 1, 3, 0.0, 2.5),
    "moderate": Tier("moderate", 3, 6, 2.5, 2.0),
    "heavy": Tier("heavy", 6, 6, 3.5, 1.0),
}


def sample_chain(
    rng: np.random.Generator,
    tier: str = "mild",
    pool: Iterable[str] | None = None,
) -> list[Cell]:
    """Sample an ordered chain of degradations for one training example."""
    tier_spec = TIERS[tier]
    if tier_spec.n_hi == 0:
        return []
    candidate_families = list(pool) if pool is not None else BRIEF_FAMILIES
    num_transforms = (
        int(rng.integers(tier_spec.n_lo, tier_spec.n_hi + 1))
        if tier_spec.n_hi > tier_spec.n_lo
        else tier_spec.n_hi
    )
    num_transforms = min(num_transforms, len(candidate_families))
    selected_families = rng.choice(
        candidate_families, size=num_transforms, replace=False
    )
    chain = []
    for name in selected_families:
        severity_level = float(
            np.clip(
                rng.normal(tier_spec.mean, tier_spec.std), SEVERITY_MIN, SEVERITY_MAX
            )
        )
        chain.append(Cell(str(name), FAMILIES[str(name)].value_at(severity_level)))
    return chain


def apply_chain(
    img: Image.Image, chain: Sequence[Cell], rng: np.random.Generator | None = None
) -> Image.Image:
    """Apply an ordered sequence of degradation cells to an image."""
    for cell in chain:
        img = cell.apply(img, rng=rng)
    return img


def tier_weights(
    epoch: int, total_epochs: int, schedule: str = "escalate"
) -> dict[str, float]:
    """Return curriculum tier probabilities for an epoch."""
    if schedule == "clean":
        return {"clean": 1.0, "mild": 0.0, "moderate": 0.0, "heavy": 0.0}
    if schedule == "fixed":
        return {"clean": 0.25, "mild": 0.25, "moderate": 0.25, "heavy": 0.25}
    if schedule != "escalate":
        raise ValueError(f"unknown schedule: {schedule}")
    progress_ratio = epoch / max(1, total_epochs - 1)  # 0 -> 1 over training
    return {
        "clean": max(0.10, 0.40 - 0.30 * progress_ratio),
        "mild": 0.40 - 0.10 * progress_ratio,
        "moderate": 0.15 + 0.20 * progress_ratio,
        "heavy": 0.05 + 0.20 * progress_ratio,
    }


def pick_tier(rng: np.random.Generator, weights: dict[str, float]) -> str:
    """Sample a curriculum tier name according to tier weights."""
    names = list(weights)
    normalized_probabilities = np.array([weights[n] for n in names], dtype=np.float64)
    normalized_probabilities = normalized_probabilities / normalized_probabilities.sum()
    return str(rng.choice(names, p=normalized_probabilities))
