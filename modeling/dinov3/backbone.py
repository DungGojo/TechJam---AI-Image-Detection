"""Backbone wrapper for pooled image embeddings."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError("timm is required: pip install 'timm>=1.0.9'") from exc


PRESETS: dict[str, dict] = {
    "dinov3_l": {"timm": "vit_large_patch16_dinov3.lvd1689m", "params": 300_000_000},
    "dinov3_b": {"timm": "vit_base_patch16_dinov3.lvd1689m", "params": 86_000_000},
    "dinov3_s": {"timm": "vit_small_patch16_dinov3.lvd1689m", "params": 21_000_000},
    "dinov2_l": {"timm": "vit_large_patch14_dinov2.lvd142m", "params": 300_000_000},
    "dinov2_b": {"timm": "vit_base_patch14_dinov2.lvd142m", "params": 86_000_000},
    "clip_l": {"timm": "vit_large_patch14_clip_224.openai", "params": 304_000_000},
    "convnext_l": {"timm": "convnext_large.fb_in22k", "params": 198_000_000},
}

PARAM_CAP = 2_000_000_000


@dataclass
class BackboneConfig:
    name: str = "dinov3_l"
    pretrained: bool = True
    pooling: str = "attention"
    freeze: bool = True
    lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_targets: list[str] = field(
        default_factory=lambda: ["qkv", "proj", "fc1", "fc2"]
    )
    img_size: int | None = None


class AttentionPool(nn.Module):
    """Pool image tokens with one learned attention query: (B, N, D) -> (B, hidden)."""

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, hidden)

    def forward(self, input_tokens: Tensor) -> Tensor:
        """(B, N, D) -> (B, hidden) pooled embedding."""
        query_expanded = self.query.expand(input_tokens.size(0), -1, -1)
        attn_out, _ = self.attn(
            query_expanded, input_tokens, input_tokens, need_weights=False
        )
        return self.proj(self.norm(attn_out.squeeze(1)))


class Backbone(nn.Module):
    """Vision Transformer or ConvNeXt feature extraction wrapper with configurable pooling."""

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        spec = PRESETS.get(cfg.name)
        timm_name = spec["timm"] if spec else cfg.name
        kwargs = {"pretrained": cfg.pretrained, "num_classes": 0}
        # img_size is ViT-only.
        if cfg.img_size is not None and "vit" in timm_name:
            kwargs["img_size"] = cfg.img_size
        self.net = timm.create_model(timm_name, **kwargs)
        self.dim = int(getattr(self.net, "num_features", 0)) or self._infer_dim()
        self.n_prefix = int(getattr(self.net, "num_prefix_tokens", 0))
        self.is_conv = "convnext" in timm_name

        if cfg.lora:
            self._apply_lora()
        elif cfg.freeze:
            for param in self.net.parameters():
                param.requires_grad_(False)
            self.net.eval()

        self.pool: nn.Module | None = (
            AttentionPool(self.dim) if cfg.pooling == "attention" else None
        )

    def _infer_dim(self) -> int:
        with torch.no_grad():
            dummy_features = self.net.forward_features(torch.zeros(1, 3, 224, 224))
        return int(dummy_features.shape[-1])

    @staticmethod
    def available() -> list[str]:
        """List names of predefined model presets."""
        return sorted(PRESETS)

    def _apply_lora(self) -> None:
        """Freeze the backbone and enable low-rank linear adapters."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            warnings.warn(
                "peft not installed -- falling back to a fully frozen backbone. "
                "pip install peft to enable the LoRA recipes.",
                stacklevel=2,
            )
            for param in self.net.parameters():
                param.requires_grad_(False)
            return
        for param in self.net.parameters():
            param.requires_grad_(False)
        self.net = get_peft_model(
            self.net,
            LoraConfig(
                r=self.cfg.lora_rank,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=0.05,
                bias="none",
                target_modules=self.cfg.lora_targets,
            ),
        )

    def _forward_features(self, x: Tensor) -> Tensor:
        """forward_features, reaching through a PEFT wrapper if LoRA is active."""
        net = self.net
        for _ in range(3):
            if hasattr(net, "forward_features"):
                return net.forward_features(x)
            net = getattr(net, "base_model", None) or getattr(net, "model", None)
            if net is None:
                break
        raise AttributeError("backbone exposes no forward_features")

    def tokens(self, x: Tensor) -> Tensor:
        """(B, 3, H, W) -> (B, N, D) patch tokens, prefix tokens stripped."""
        raw_features = self._forward_features(x)
        if self.is_conv:  # (B, D, h, w) -> (B, hw, D)
            return raw_features.flatten(2).transpose(1, 2)
        if raw_features.ndim == 2:  # already pooled
            return raw_features.unsqueeze(1)
        return raw_features[:, self.n_prefix :, :] if self.n_prefix else raw_features

    def forward(self, x: Tensor) -> Tensor:
        """(B, 3, H, W) -> (B, D) pooled embedding."""
        mode = self.cfg.pooling
        if mode == "cls" and not self.is_conv and self.n_prefix:
            return self._forward_features(x)[:, 0, :]
        patch_tokens = self.tokens(x)
        if mode == "mean":
            return patch_tokens.mean(dim=1)
        if mode == "max":
            return patch_tokens.max(dim=1).values
        if mode == "attention":
            if self.pool is None:
                raise RuntimeError("AttentionPool layer is uninitialized")
            return self.pool(patch_tokens)
        if mode == "cls":
            # Fallback for models without dedicated prefix CLS tokens (e.g. ConvNeXt)
            return patch_tokens.mean(dim=1)
        raise ValueError(f"unknown pooling: {mode}")

    def param_counts(self) -> tuple[int, int]:
        """Return (total_params, trainable_params)."""
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )
        return total, trainable


def assert_within_cap(*modules: nn.Module) -> int:
    """Hard gate on the competition's 2B limit. Call this in train and predict entrypoints."""
    total = sum(param.numel() for m in modules for param in m.parameters())
    if total > PARAM_CAP:
        raise RuntimeError(
            f"inference footprint is {total:,} parameters, over the {PARAM_CAP:,} cap"
        )
    return total
