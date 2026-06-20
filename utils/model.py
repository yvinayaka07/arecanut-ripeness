"""
utils/model.py
==============
DINOv2-based classifier for arecanut ripeness (ripe / unripe).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("arecanut.model")

# DINOv2 variant → embedding dimension
EMBED_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ArecanutRipenessClassifier(nn.Module):
    """
    DINOv2 backbone + MLP head for binary ripeness classification.

    Phase 1: backbone frozen, train head only.
    Phase 2: unfreeze backbone for fine-tuning.
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vits14",
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self._backbone_frozen = freeze_backbone

        logger.info(f"Loading DINOv2 backbone: {backbone_name}")
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_name, pretrained=True
        )

        embed_dim = EMBED_DIMS.get(backbone_name, 384)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Init head weights
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._backbone_frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            return probs.argmax(dim=-1), probs


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: ArecanutRipenessClassifier,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    checkpoint_dir: str | Path,
    is_best: bool = False,
) -> None:
    """Save training checkpoint. If is_best, also write best_model.pth."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "backbone_name": model.backbone_name,
        "num_classes": model.num_classes,
    }

    path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
    torch.save(state, path)
    logger.info(f"Checkpoint saved: {path}")

    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(state, best_path)
        logger.info(f"Best model updated: {best_path}")


def load_checkpoint(
    checkpoint_path: str | Path,
    model: ArecanutRipenessClassifier,
    optimizer=None,
    scheduler=None,
    device: Optional[torch.device] = None,
) -> dict:
    """Load checkpoint weights into model (and optionally optimizer/scheduler)."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device or torch.device("cpu"))
    model.load_state_dict(state["model_state_dict"])
    logger.info(f"Loaded weights from: {checkpoint_path} (epoch {state['epoch']})")

    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler and state.get("scheduler_state_dict"):
        scheduler.load_state_dict(state["scheduler_state_dict"])

    return {"epoch": state.get("epoch", 0), "metrics": state.get("metrics", {})}


def build_model_from_config(config, device: torch.device) -> ArecanutRipenessClassifier:
    """Build model from config and move to device."""
    cls_cfg = config.classification
    model = ArecanutRipenessClassifier(
        backbone_name=cls_cfg.backbone,
        num_classes=config.classes.num_classes,
        hidden_dim=cls_cfg.head.hidden_dim,
        dropout=cls_cfg.head.dropout,
        freeze_backbone=cls_cfg.freeze_backbone,
    )
    return model.to(device)
