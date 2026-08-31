"""WildFake archive metadata and generator parsing."""

from __future__ import annotations

import json
from pathlib import Path

from modelscope.hub.api import HubApi

from Data_Pipeline.sources.remote_zip import RemoteZip, modelscope_url

REPO_ID = "hy2628982280/WildFake"
REVISION = "master"

TREE_CACHE = Path("data/train/images/wildfake_meta/file_tree.json")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

DEMO_REAL = {
    "archive": "Images/Real/coco.zip",
    "prefix": "coco/coco2017/val2017/",
    "expected": 4_998,
    "label": 0,
    "generator": "real",
    "subdir": "real/coco_val2017",
}
DEMO_FAKE = {
    "archive": "Images/Diffusion_based/DALLE.zip",
    "prefix": "DALLE/Advanced/DALLE3/",
    "expected": 8_843,
    "label": 1,
    "generator": "dalle3",
    "subdir": "fake/dalle3_advanced",
}
DEMO_TOTAL = DEMO_REAL["expected"] + DEMO_FAKE["expected"]


def file_tree(refresh: bool = False) -> list[dict]:
    """Every blob in the repo, with sizes. Cached to disk."""
    if TREE_CACHE.exists() and not refresh:
        return json.loads(TREE_CACHE.read_text())

    api = HubApi()
    entries: list[dict] = []
    page = 1
    while True:
        batch = api.get_dataset_files(
            repo_id=REPO_ID, revision=REVISION, recursive=True,
            page_number=page, page_size=100,
        )
        if not batch:
            break
        entries += batch
        if len(batch) < 100:
            break
        page += 1

    TREE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TREE_CACHE.write_text(json.dumps(entries, indent=1))
    return entries


def archive_sizes(refresh: bool = False) -> dict[str, int]:
    """Return the exact byte size of each archive."""
    return {
        f["Path"]: (f.get("Size") or 0)
        for f in file_tree(refresh)
        if f.get("Type") != "tree"
    }


def open_archive(path: str, sizes: dict[str, int] | None = None) -> RemoteZip:
    sizes = sizes if sizes is not None else archive_sizes()
    if path not in sizes:
        raise KeyError(f"{path} is not in the WildFake repo")
    return RemoteZip(modelscope_url(REPO_ID, path, REVISION), sizes[path])


def image_members(zip_handle: RemoteZip, prefix: str = "") -> list:
    """Members under `prefix` that are actually images."""
    return [
        m
        for m in zip_handle.members()
        if not m.is_dir
        and m.name.startswith(prefix)
        and m.name.lower().endswith(IMAGE_SUFFIXES)
    ]


# Generator identity may come from a tiered member path or the archive root.
TIERS = {"Advanced", "Typical"}

SUBGENERATOR_ROOTS = {"personalizedSD", "SDwithAdaptor"}


def generator_from_archive(archive_path: str) -> str:
    """Infer generator identity from an archive path."""
    parts = Path(archive_path).parts
    stem = Path(archive_path).stem
    if stem.startswith("part_") and len(parts) >= 3:
        family, tier = parts[-3], parts[-2]
        return f"{family}_{tier}" if tier in TIERS else family
    return stem


def generator_of(name: str, archive_path: str = "") -> tuple[str, bool]:
    """Infer generator identity from a member, then its archive."""
    parts = name.split("/")
    if len(parts) >= 3 and parts[1] in TIERS:
        return parts[2], parts[1] == "Advanced"
    if len(parts) >= 2 and parts[0] in SUBGENERATOR_ROOTS:
        return f"{parts[0]}_{parts[1]}", False
    if archive_path:
        generator = generator_from_archive(archive_path)
        return generator, generator.endswith("_Advanced")
    return parts[0], False


# One large part per generator is sufficient for the configured sample cap.
FAKE_ARCHIVES = [
    "Images/Diffusion_based/DALLE.zip",
    "Images/Diffusion_based/DDIM.zip",
    "Images/Diffusion_based/DDPM.zip",
    "Images/Diffusion_based/ADM.zip",
    "Images/Diffusion_based/VQDM.zip",
    "Images/Diffusion_based/Imagen.zip",
    "Images/Other_based.zip",
    "Images/GAN_based.zip",
    "Images/Diffusion_based/SD/personalizedSD.zip",
    "Images/Diffusion_based/SD/SDwithAdaptor.zip",
    "Images/Diffusion_based/SD/originalSD/Typical/part_1.zip",
    "Images/Diffusion_based/SD/originalSD/Advanced/part_1.zip",
    "Images/Diffusion_based/Midjourney/Typical/part_1.zip",
    "Images/Diffusion_based/Midjourney/Advanced/part_1.zip",
]

REAL_ARCHIVES = [
    "Images/Real/coco.zip",
    "Images/Real/imagenet.zip",
    "Images/Real/ffhq.zip",
    "Images/Real/church.zip",
    "Images/Real/afhq.zip",
    "Images/Real/celebahq.zip",
]

# Exclude quarantined evaluation paths before downloading.
QUARANTINED_PREFIXES = (
    "coco/coco2017/val2017/",
    "DALLE/Advanced/DALLE3/",
)


def is_quarantined(name: str) -> bool:
    return any(name.startswith(p) for p in QUARANTINED_PREFIXES)


# These generator families appear only in the unseen test split.
UNSEEN_GENERATORS = (
    "GigaGAN",
    "GALIP",
    "MAGE",
    "SDwithAdaptor_lycris",
    "Midjourney_Advanced",
)
