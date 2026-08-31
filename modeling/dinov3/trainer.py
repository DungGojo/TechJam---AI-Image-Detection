"""DINOv3 training loop with robust selection and resumable checkpoints."""

from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_pipeline.datasets import DataConfig, PairedManifestDataset
from evaluation.training_validation import evaluate_robust, summarise
from modeling.common.device import get_device
from modeling.dinov3.losses import LossWeights, RobustLoss


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-5
    weight_decay: float = 0.02
    warmup_epochs: float = 1.0
    grad_clip: float = 1.0
    amp: bool = True
    ema: bool = False
    ema_decay: float = 0.999
    swa: bool = False
    swa_start_frac: float = 0.75
    num_workers: int = 4
    seed: int = 0
    out_dir: str = "outputs/runs"
    checkpoints_dir: str = "checkpoints/dinov3"
    resume: str | None = None
    save_every: int = 1
    progress_interval_seconds: float = 300.0
    eval_every: int = 1
    max_eval_images: int = 2000
    loss: LossWeights = field(default_factory=LossWeights)


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            key: value.detach().clone().float()
            for key, value in model.state_dict().items()
            if value.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            if key in self.shadow:
                self.shadow[key].mul_(self.decay).add_(
                    value.detach().float(), alpha=1 - self.decay
                )

    def copy_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        backup = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if key in self.shadow
        }
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)
        return backup

    @staticmethod
    def restore(model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        model.load_state_dict({**model.state_dict(), **backup}, strict=False)


def build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    warmup = max(1, int(cfg.warmup_epochs * steps_per_epoch))
    total = max(warmup + 1, cfg.epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _save_resume_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    ema: EMA | None,
    swa_model,
    epoch: int,
    step: int,
    history: list,
    best: dict,
    data_cfg: DataConfig,
    cfg: TrainConfig,
    use_amp: bool,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "ema": ema.shadow if ema is not None else None,
            "swa": swa_model.state_dict() if swa_model is not None else None,
            "epoch": epoch,
            "step": step,
            "history": history,
            "best": best,
            "data": asdict(data_cfg),
            "train": asdict(cfg),
            "model_config": asdict(model.cfg),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
        path,
    )


def train(
    model: nn.Module, data_cfg: DataConfig, cfg: TrainConfig, val_df=None
) -> dict:
    """Train one expert; resume safely across a 12-hour cluster allocation."""
    device = get_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model.to(device)

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"device={device}  params={total / 1e6:.1f}M  trainable={trainable / 1e6:.2f}M"
    )
    if total > 2_000_000_000:
        raise RuntimeError(f"{total:,} parameters exceeds the 2B competition cap")
    if trainable == 0:
        raise RuntimeError("nothing is trainable; check freeze/lora settings")

    dataset = PairedManifestDataset(data_cfg, split="train")
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg, len(loader))
    criterion = RobustLoss(cfg.loss).to(device)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, cfg.ema_decay) if cfg.ema else None
    swa_model = torch.optim.swa_utils.AveragedModel(model) if cfg.swa else None
    swa_start = int(cfg.swa_start_frac * cfg.epochs)

    run_dir = Path(cfg.out_dir) / time.strftime("%Y%m%d-%H%M%S")
    checkpoints_dir = Path(cfg.checkpoints_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {"train": asdict(cfg), "data": asdict(data_cfg)}, indent=2, default=str
        )
    )

    history: list = []
    best = {"robust_auc": -1.0, "epoch": -1}
    step, start_epoch = 0, 0
    total_steps = max(1, cfg.epochs * len(loader))
    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if use_amp and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        if ema is not None and checkpoint.get("ema"):
            ema.shadow = checkpoint["ema"]
        if swa_model is not None and checkpoint.get("swa"):
            swa_model.load_state_dict(checkpoint["swa"])
        history = checkpoint.get("history", [])
        best = checkpoint.get("best", best)
        step = int(checkpoint.get("step", 0))
        start_epoch = int(checkpoint["epoch"]) + 1
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if (
            torch.cuda.is_available()
            and checkpoint.get("cuda_rng_state_all") is not None
        ):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        print(f"resumed {cfg.resume} at epoch {start_epoch}, step {step}")

    progress = tqdm(
        total=total_steps,
        initial=min(step, total_steps),
        desc="Training",
        unit="batch",
        mininterval=cfg.progress_interval_seconds,
        maxinterval=cfg.progress_interval_seconds,
        miniters=0,
        dynamic_ncols=True,
        leave=True,
    )
    for epoch in range(start_epoch, cfg.epochs):
        dataset.set_epoch(epoch)
        epoch_totals: dict[str, float] = {}
        epoch_batches = 0
        for batch in loader:
            if data_cfg.pairwise:
                clean, degraded, labels = batch
                clean, degraded, labels = (
                    clean.to(device),
                    degraded.to(device),
                    labels.to(device),
                )
            else:
                clean, labels = batch
                clean, labels, degraded = clean.to(device), labels.to(device), None

            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.amp.autocast("cuda", enabled=True)
                if use_amp
                else contextlib.nullcontext()
            )
            with amp_context:
                clean_logits, clean_features = model(clean)
                if degraded is not None:
                    degraded_logits, degraded_features = model(degraded, degraded=True)
                    loss, parts = criterion(
                        clean_logits,
                        labels,
                        degraded_logits,
                        clean_features,
                        degraded_features,
                    )
                else:
                    loss, parts = criterion(clean_logits, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            for key, value in parts.items():
                epoch_totals[key] = epoch_totals.get(key, 0.0) + value
            epoch_batches += 1
            step += 1
            progress.set_postfix(
                epoch=f"{epoch + 1}/{cfg.epochs}",
                loss=f"{parts['total']:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                refresh=False,
            )
            progress.update(1)
            if progress.n >= progress.total:
                progress.refresh()

        train_stats = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            **{
                f"train_{key}": value / max(1, epoch_batches)
                for key, value in epoch_totals.items()
            },
        }
        epoch_record: dict = {
            "train": train_stats,
            "summary": {"epoch": epoch},
            "grid": {},
        }

        if swa_model is not None and epoch >= swa_start:
            swa_model.update_parameters(model)

        if val_df is not None and (epoch + 1) % cfg.eval_every == 0:
            backup = ema.copy_to(model) if ema is not None else None
            grid = evaluate_robust(
                model,
                val_df,
                data_cfg.input_size,
                data_cfg.crop_mode,
                device,
                max_images=cfg.max_eval_images,
                seed=cfg.seed,
                progress_interval_seconds=cfg.progress_interval_seconds,
            )
            if ema is not None and backup is not None:
                ema.restore(model, backup)
            summary = summarise(grid)
            summary["epoch"] = epoch
            epoch_record.update({"summary": summary, "grid": grid})
            tqdm.write(
                f"[epoch {epoch}] clean={summary['clean_auc']:.4f} "
                f"robust={summary['robust_auc']:.4f} "
                f"worst={summary['worst_cell_auc']:.4f} gap={summary['gap']:.4f} "
                f"loss={train_stats.get('train_total', float('nan')):.4f}"
            )
            if summary["robust_auc"] > best["robust_auc"]:
                best = dict(summary)
                inference_state = model.state_dict()
                if ema is not None:
                    inference_state = {**inference_state, **ema.shadow}
                torch.save(
                    {
                        "model": inference_state,
                        "model_config": asdict(model.cfg),
                        "epoch": epoch,
                        "summary": summary,
                    },
                    checkpoints_dir / "best_robust.pt",
                )
                tqdm.write("           new best robust AUC; checkpoint saved")

        history.append(epoch_record)

        if (epoch + 1) % cfg.save_every == 0:
            _save_resume_checkpoint(
                checkpoints_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                swa_model=swa_model,
                epoch=epoch,
                step=step,
                history=history,
                best=best,
                data_cfg=data_cfg,
                cfg=cfg,
                use_amp=use_amp,
            )

    progress.close()

    if swa_model is not None:
        try:
            torch.optim.swa_utils.update_bn(loader, swa_model, device=device)
        except Exception as exc:
            print(f"SWA batch-norm update skipped: {exc}")
        torch.save({"model": swa_model.module.state_dict()}, checkpoints_dir / "swa.pt")

    (run_dir / "history.json").write_text(json.dumps(history, indent=2, default=str))
    print(f"run metadata written to {run_dir}")
    print(f"model checkpoints written to {checkpoints_dir}")
    return {"best": best, "history": history, "out_dir": str(run_dir)}
