"""Expert prediction fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor


def weighted_mean(probs: Sequence[Tensor], weights: Sequence[float] | None = None) -> Tensor:
    """Compute weighted average of probability tensors: (M, B) -> (B,)."""
    stacked_probs = torch.stack([t.float() for t in probs], dim=0)          # (M, B)
    if weights is None:
        return stacked_probs.mean(dim=0)
    normalized_weights = torch.tensor(weights, dtype=stacked_probs.dtype, device=stacked_probs.device).view(-1, 1)
    normalized_weights = normalized_weights / normalized_weights.sum()
    return (stacked_probs * normalized_weights).sum(dim=0)


def logit_mean(logits: Sequence[Tensor], weights: Sequence[float] | None = None) -> Tensor:
    """Compute sigmoid of weighted logit average: (M, B) -> (B,)."""
    stacked_logits = torch.stack([t.float() for t in logits], dim=0)
    if weights is None:
        return torch.sigmoid(stacked_logits.mean(dim=0))
    normalized_weights = torch.tensor(weights, dtype=stacked_logits.dtype, device=stacked_logits.device).view(-1, 1)
    return torch.sigmoid((stacked_logits * (normalized_weights / normalized_weights.sum())).sum(dim=0))


# Weighted nodes preserve hierarchical fusion groupings.

@dataclass
class Node:
    """Either a leaf (an expert name) or a weighted group of child nodes."""
    weight: float = 1.0
    expert: str | None = None
    children: list["Node"] = field(default_factory=list)

    def evaluate(self, values: dict[str, Tensor]) -> Tensor:
        """Recursively evaluate the hierarchical node tree given expert predictions."""
        if self.expert is not None:
            if self.expert not in values:
                raise KeyError(f"no prediction for expert '{self.expert}'")
            return values[self.expert].float()
        if not self.children:
            raise ValueError("node has neither an expert nor children")
        total_child_weight = sum(c.weight for c in self.children)
        out = None
        for child in self.children:
            term = child.evaluate(values) * (child.weight / total_child_weight)
            out = term if out is None else out + term
        return out


def intsig_tree() -> Node:
    """The published 4th-place structure, for reference and for reproduction runs."""
    return Node(children=[
        Node(weight=0.7, children=[
            Node(weight=0.7, children=[
                Node(weight=0.75, expert="m1"),
                Node(weight=0.15, expert="m2"),
                Node(weight=0.10, expert="m3"),
            ]),
            Node(weight=0.3, expert="m4"),
        ]),
        Node(weight=0.3, expert="m5"),
    ])


def tree_from_config(spec: dict) -> Node:
    """Parse a hierarchical tree specification from dict."""
    if "expert" in spec:
        return Node(weight=float(spec.get("weight", 1.0)), expert=str(spec["expert"]))
    return Node(
        weight=float(spec.get("weight", 1.0)),
        children=[tree_from_config(c) for c in spec.get("children", [])],
    )



@dataclass
class GateConfig:
    enabled: bool = False
    strong_experts: tuple[str, str] = ("m4", "m5")
    strong_thresholds: tuple[float, float] = (8.0, 3.0)
    shift: float = 2.5
    majority_experts: tuple[str, ...] = ("m1", "m2", "m3", "m5")
    majority_min: int = 3
    dissenter: str = "m4"


def _direction(logit: Tensor) -> Tensor:
    return torch.sign(logit)


def apply_dual_gate(
    fused_logit: Tensor,
    expert_logits: dict[str, Tensor],
    tree: Node,
    cfg: GateConfig,
) -> Tensor:
    """Correct fused logits when specialist agreement passes a confidence gate."""
    if not cfg.enabled:
        return fused_logit
    out = fused_logit.clone()

    expert_a, expert_b = cfg.strong_experts
    if expert_a in expert_logits and expert_b in expert_logits:
        threshold_a, threshold_b = cfg.strong_thresholds
        is_strong = (expert_logits[expert_a].abs() >= threshold_a) & (expert_logits[expert_b].abs() >= threshold_b)
        is_agree = _direction(expert_logits[expert_a]) == _direction(expert_logits[expert_b])
        is_contradict = _direction(expert_logits[expert_a]) != _direction(out)
        mask = is_strong & is_agree & is_contradict
        out = torch.where(mask, out + cfg.shift * _direction(expert_logits[expert_a]), out)

    present = [e for e in cfg.majority_experts if e in expert_logits]
    if cfg.dissenter in expert_logits and len(present) >= cfg.majority_min:
        expert_directions = torch.stack([_direction(expert_logits[e]) for e in present], dim=0)
        modal = torch.sign(expert_directions.sum(dim=0))
        num_agreeing = (expert_directions == modal.unsqueeze(0)).sum(dim=0)
        dissents = _direction(expert_logits[cfg.dissenter]) != modal
        mask = (num_agreeing >= cfg.majority_min) & dissents
        if mask.any():
            without = {k: v for k, v in expert_logits.items() if k != cfg.dissenter}
            try:
                refused = tree.evaluate(without)
                out = torch.where(mask, refused, out)
            except KeyError:
                pass  # the dissenter is load-bearing in this tree; leave the fusion alone
    return out



@dataclass
class FusionConfig:
    mode: str = "mean"
    weights: dict[str, float] = field(default_factory=dict)
    tree: dict | None = None
    gate: GateConfig = field(default_factory=GateConfig)


def fuse(
    expert_probs: dict[str, Tensor],
    expert_logits: dict[str, Tensor] | None = None,
    cfg: FusionConfig | None = None,
) -> Tensor:
    """Combine per-expert predictions into a single probability per image."""
    cfg = cfg or FusionConfig()
    names = list(expert_probs)

    if cfg.mode == "mean":
        return weighted_mean([expert_probs[n] for n in names])
    if cfg.mode == "weighted":
        weights_list = [cfg.weights.get(n, 1.0) for n in names]
        return weighted_mean([expert_probs[n] for n in names], weights_list)
    if cfg.mode == "logit":
        if expert_logits is None:
            raise ValueError("logit fusion needs expert_logits")
        weights_list = [cfg.weights.get(n, 1.0) for n in names] if cfg.weights else None
        return logit_mean([expert_logits[n] for n in names], weights_list)
    if cfg.mode == "hierarchical":
        tree = tree_from_config(cfg.tree) if cfg.tree else intsig_tree()
        if cfg.gate.enabled:
            if expert_logits is None:
                raise ValueError("gating operates on logits; pass expert_logits")
            fused_logit = tree.evaluate(expert_logits)
            return torch.sigmoid(apply_dual_gate(fused_logit, expert_logits, tree, cfg.gate))
        return tree.evaluate(expert_probs)
    raise ValueError(f"unknown fusion mode: {cfg.mode}")

