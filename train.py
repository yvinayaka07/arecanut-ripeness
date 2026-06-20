"""
train.py
========
Train a DINOv2 classifier for arecanut ripeness detection.

Usage:
    python train.py                       # fresh training
    python train.py --resume models/checkpoints/best_model.pth
    python train.py --epochs 30 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent))
from utils.common import load_config, setup_logger, get_device, seed_everything
from utils.dataset import get_dataloaders, validate_dataset_structure
from utils.model import build_model_from_config, save_checkpoint, load_checkpoint


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv2 ripeness classifier")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, epoch, logger):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += images.size(0)

        if i % 10 == 0:
            logger.info(f"  Epoch {epoch} [{i}/{len(loader)}] loss={loss.item():.4f}")

    return {"train_loss": total_loss / total, "train_acc": correct / total}


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    return {
        "val_loss": total_loss / n,
        "val_acc": accuracy_score(all_labels, all_preds),
        "val_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "val_precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "val_recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    logger = setup_logger("arecanut.train")
    seed_everything(args.seed)

    # CLI overrides
    cls_cfg = config.classification.training
    if args.epochs:
        cls_cfg.epochs = args.epochs
    if args.batch_size:
        cls_cfg.batch_size = args.batch_size
    if args.lr:
        cls_cfg.learning_rate = args.lr

    device = get_device(args.device or config.detection.device)
    logger.info(f"Device: {device}")

    # Validate dataset
    logger.info("Validating dataset structure...")
    if not validate_dataset_structure(config.paths.dataset_root):
        logger.error("Dataset validation failed. Ensure dataset/ has train/val/test with ripe/ and unripe/ subdirs.")
        sys.exit(1)

    # DataLoaders
    dataloaders = get_dataloaders(config)

    # Model
    model = build_model_from_config(config, device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=cls_cfg.label_smoothing)

    if config.classification.freeze_backbone:
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=cls_cfg.learning_rate, weight_decay=cls_cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": cls_cfg.learning_rate * cls_cfg.backbone_lr_multiplier},
            {"params": model.head.parameters(), "lr": cls_cfg.learning_rate},
        ], weight_decay=cls_cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cls_cfg.epochs, eta_min=1e-6
    )

    use_amp = device.type == "cuda" and config.hardware.mixed_precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        meta = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = meta["epoch"] + 1

    Path(config.paths.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    class_names = config.classes.names
    epochs = cls_cfg.epochs
    patience = cls_cfg.patience
    unfreeze_after = config.classification.unfreeze_after_epoch

    best_f1 = 0.0
    no_improve = 0
    history = []

    logger.info(f"Starting training | Epochs: {epochs} | Device: {device} | AMP: {use_amp}")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # Unfreeze backbone for fine-tuning at the configured epoch
        if epoch == unfreeze_after and model._backbone_frozen:
            logger.info(f"Epoch {epoch}: Unfreezing backbone for fine-tuning.")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": cls_cfg.learning_rate * cls_cfg.backbone_lr_multiplier},
                {"params": model.head.parameters(), "lr": cls_cfg.learning_rate},
            ], weight_decay=cls_cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - epoch, eta_min=1e-6)

        train_m = train_one_epoch(model, dataloaders["train"], optimizer, criterion, device, scaler, epoch, logger)
        val_m = evaluate(model, dataloaders["val"], criterion, device, class_names)
        scheduler.step()

        is_best = val_m["val_f1"] > best_f1
        if is_best:
            best_f1 = val_m["val_f1"]
            no_improve = 0
        else:
            no_improve += 1

        epoch_metrics = {**train_m, **val_m, "epoch": epoch}
        history.append(epoch_metrics)

        logger.info(
            f"Epoch [{epoch:03d}/{epochs-1}] "
            f"loss={train_m['train_loss']:.4f} acc={train_m['train_acc']:.4f} | "
            f"val_loss={val_m['val_loss']:.4f} val_acc={val_m['val_acc']:.4f} val_f1={val_m['val_f1']:.4f} "
            f"{'[BEST]' if is_best else ''} | {time.time()-t0:.1f}s"
        )

        save_checkpoint(model, optimizer, scheduler, epoch, epoch_metrics,
                        config.paths.checkpoints_dir, is_best=is_best)

        if no_improve >= patience:
            logger.info(f"Early stopping — no improvement for {patience} epochs.")
            break

    # Save training history
    Path(config.paths.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    hist_path = Path(config.paths.checkpoints_dir) / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training complete. Best val_f1: {best_f1:.4f} | History: {hist_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
