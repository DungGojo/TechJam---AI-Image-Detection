#!/usr/bin/env python3
"""Train a detector recipe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.datasets import DataConfig
from data_pipeline.manifest import load as load_manifest
from modeling.dinov3.backbone import BackboneConfig
from modeling.dinov3.detector import Expert, ExpertConfig
from modeling.dinov3.losses import LossWeights
from modeling.dinov3.trainer import TrainConfig, train


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    return value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """--override epochs=4 model.backbone.pooling=mean"""
    for item in overrides:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(raw)
    return cfg


def build_expert(spec: dict) -> Expert:
    bb = BackboneConfig(**spec.get("backbone", {}))
    fields = {k: v for k, v in spec.items() if k != "backbone"}
    return Expert(ExpertConfig(backbone=bb, **fields))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True, type=Path)
    ap.add_argument(
        "--manifest", default=None, help="overrides data.manifest in the recipe"
    )
    ap.add_argument("--val-manifest", default=None)
    ap.add_argument("--resume", default=None, help="resume from a last.pt checkpoint")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.recipe.read_text())
    cfg = apply_overrides(cfg, args.override)
    if args.manifest:
        cfg.setdefault("data", {})["manifest"] = args.manifest

    print(f"recipe: {cfg.get('name', args.recipe.stem)}")
    if cfg.get("reproduces"):
        print(f"reproduces: {cfg['reproduces']}")
    if cfg.get("scaled_down"):
        print(f"scaled down: {cfg['scaled_down']}")

    experts_spec = cfg.get("experts") or [cfg.get("model", {})]
    if len(experts_spec) > 1:
        print(
            f"\nthis recipe defines {len(experts_spec)} experts; training them one at a "
            f"time.\nfuse the trained checkpoints afterwards with modeling/ensemble/fusion.py\n"
        )

    if args.dry_run:
        return dry_run(experts_spec, cfg)

    train_cfg = TrainConfig(
        loss=LossWeights(**cfg.get("loss", {})),
        **{k: v for k, v in cfg.get("train", {}).items()},
    )
    if args.resume:
        train_cfg.resume = args.resume
    val_df = None
    val_path = args.val_manifest or cfg.get("data", {}).get("val_manifest")
    if val_path:
        vdf = load_manifest(val_path)
        val_df = vdf[vdf["split"] == "val"] if "split" in vdf.columns else vdf
        print(f"validation: {len(val_df)} images from {val_path}")
    else:
        print("no validation manifest -- checkpoint selection is disabled")

    for i, spec in enumerate(experts_spec):
        name = spec.get("name", f"expert{i}")
        print(f"\n{'=' * 70}\nexpert {i + 1}/{len(experts_spec)}: {name}\n{'=' * 70}")
        model = build_expert({k: v for k, v in spec.items() if k != "name"})
        print(model.describe())
        data_kwargs = {
            k: v
            for k, v in cfg.get("data", {}).items()
            if k not in {"val_manifest", "input_size", "crop_mode", "total_epochs"}
        }
        data_cfg = DataConfig(
            input_size=model.cfg.input_size,
            crop_mode=model.cfg.crop_mode,
            total_epochs=train_cfg.epochs,
            **data_kwargs,
        )
        train_cfg.out_dir = (
            f"{cfg.get('train', {}).get('out_dir', 'outputs/runs')}/{name}"
        )
        train_cfg.checkpoints_dir = f"checkpoints/dinov3/{name}"
        train(model, data_cfg, train_cfg, val_df=val_df)
    return 0


def dry_run(experts_spec: list[dict], cfg: dict) -> int:
    """Build every expert with random weights and run one synthetic step."""
    import torch

    from modeling.dinov3.losses import RobustLoss

    ok = True
    for i, spec in enumerate(experts_spec):
        name = spec.get("name", f"expert{i}")
        spec = {k: v for k, v in spec.items() if k != "name"}
        spec.setdefault("backbone", {})["pretrained"] = False
        try:
            model = build_expert(spec)
            model.train()
            size = model.cfg.input_size
            x = torch.randn(2, 3, size, size)
            y = torch.tensor([0.0, 1.0])
            logit_c, feat_c = model(x)
            logit_d, feat_d = model(x, degraded=True)
            loss, parts = RobustLoss(LossWeights(**cfg.get("loss", {})))(
                logit_c, y, logit_d, feat_c, feat_d
            )
            loss.backward()
            grads = sum(
                int(p.grad is not None) for p in model.parameters() if p.requires_grad
            )
            total, trainable = model.param_counts()
            print(f"  ok  {name}: {model.describe()}")
            print(
                f"      logits {tuple(logit_c.shape)}  feats {tuple(feat_c.shape)}  "
                f"loss={parts['total']:.4f}  tensors_with_grad={grads}"
            )
            if total > 2_000_000_000:
                print(f"      OVER CAP: {total:,} parameters")
                ok = False
            if trainable == 0:
                print("      nothing trainable -- check freeze/lora settings")
                ok = False
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print("\ndry run passed" if ok else "\ndry run FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
