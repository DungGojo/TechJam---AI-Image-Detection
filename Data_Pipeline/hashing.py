"""Content and perceptual image hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dct

_CHUNK = 1 << 20

# Conventional 64-bit pHash threshold that survives compression and resizing.
PHASH_MATCH_THRESHOLD = 10


def sha256_file(path: str | Path) -> str:
    """Streamed sha256 of the file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """Return a 64-bit DCT perceptual hash as 16 hexadecimal characters."""
    img_size = hash_size * highfreq_factor
    grey = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(grey, dtype=np.float64)

    coeffs = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coeffs[:hash_size, :hash_size]
    # Exclude overall brightness from the pHash threshold.
    median = np.median(low.flatten()[1:])
    bits = (low > median).flatten()

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def phash_file(path: str | Path) -> str:
    from Data_Pipeline.image_io import load_image

    return phash(load_image(path))


def hamming(a: str, b: str) -> int:
    """Bit distance between two hex hashes of equal width."""
    if len(a) != len(b):
        raise ValueError(f"hash width mismatch: {len(a)} vs {len(b)}")
    return int(int(a, 16) ^ int(b, 16)).bit_count()


class PHashIndex:
    """Look up near-duplicates by perceptual-hash distance."""

    def __init__(self, hashes: list[str]):
        self.hashes = list(hashes)
        self._values = np.array([int(h, 16) for h in self.hashes], dtype=object)

    def __len__(self) -> int:
        return len(self.hashes)

    def closest(self, query: str) -> tuple[int, str | None]:
        """Return (distance, matching_hash) for the nearest entry."""
        if not self.hashes:
            return (10**9, None)
        q = int(query, 16)
        best_d, best_h = 10**9, None
        for value, raw in zip(self._values, self.hashes):
            d = int(value ^ q).bit_count()
            if d < best_d:
                best_d, best_h = d, raw
                if d == 0:
                    break
        return best_d, best_h

    def matches(self, query: str, threshold: int = PHASH_MATCH_THRESHOLD) -> bool:
        return self.closest(query)[0] <= threshold


def read_hash_list(path: str | Path) -> list[str]:
    """Read a blocklist file: one hash per line, '#' comments allowed."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line.split()[0].lower())
    return out


def write_hash_list(path: str | Path, hashes: list[str], header: str = "") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if header:
        lines += [f"# {ln}" for ln in header.splitlines()]
    lines += list(hashes)
    p.write_text("\n".join(lines) + "\n")
