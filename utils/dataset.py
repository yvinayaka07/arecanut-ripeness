"""
utils/dataset.py
================
Dataset classes, transform pipelines, data loading, augmentation,
class balancing, and dataset statistics for arecanut ripeness classification.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

logger = logging.getLogger("arecanut.dataset")


# ---------------------------------------------------------------------------
# Transform Pipelines
# ---------------------------------------------------------------------------

def get_train_transforms(img_size: int = 224, augmentation_config=None) -> transforms.Compose:
    """
    Build training transform pipeline with aggressive augmentation.
    Handles lighting variation critical for outdoor robot environments.
    """
    transform_list = [
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
        ),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ]
    return transforms.Compose(transform_list)


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """Deterministic val/test transforms — resize and normalize only."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_inference_transforms(img_size: int = 224) -> transforms.Compose:
    """Single-image inference transform."""
    return get_val_transforms(img_size)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization for visualization."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------------------

class ArecanutDataset(Dataset):
    """
    ImageFolder-style dataset for arecanut ripeness classification.

    Expects:
        root/
            ripe/   *.jpg *.png ...
            unripe/ *.jpg *.png ...

    Args:
        root_dir: Path to split root (train / val / test).
        transform: Torchvision transform pipeline.
        class_names: Override class ordering (default: sorted folder names).
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
        class_names: Optional[list[str]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        # Discover classes from subdirectory names
        available = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        if not available:
            raise ValueError(f"No class subdirectories found in {self.root_dir}")

        self.class_names = class_names if class_names else available
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

        # Collect (image_path, label) pairs
        self.samples: list[tuple[Path, int]] = []
        self._collect_samples()

        logger.info(
            f"Dataset loaded: {self.root_dir.name} | "
            f"{len(self.samples)} images | "
            f"Classes: {self.class_names}"
        )

    def _collect_samples(self) -> None:
        for cls_name in self.class_names:
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists():
                logger.warning(f"Class directory not found: {cls_dir}")
                continue
            label = self.class_to_idx[cls_name]
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            # Return a black image as fallback rather than crashing
            image = Image.new("RGB", (224, 224), color=0)

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_counts(self) -> dict[str, int]:
        label_counts = Counter(label for _, label in self.samples)
        return {self.class_names[k]: v for k, v in sorted(label_counts.items())}

    def get_sample_weights(self) -> list[float]:
        """Compute per-sample weights for WeightedRandomSampler (class balancing)."""
        counts = self.get_class_counts()
        class_weights = {
            self.class_to_idx[cls]: 1.0 / count
            for cls, count in counts.items()
        }
        return [class_weights[label] for _, label in self.samples]

    def summary(self) -> str:
        counts = self.get_class_counts()
        total = len(self.samples)
        lines = [f"Dataset: {self.root_dir}", f"Total: {total}"]
        for cls, count in counts.items():
            pct = 100 * count / total if total else 0
            lines.append(f"  {cls}: {count} ({pct:.1f}%)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset: ArecanutDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    balance_classes: bool = False,
) -> DataLoader:
    """
    Construct a DataLoader with optional class balancing via WeightedRandomSampler.

    Args:
        dataset: ArecanutDataset instance.
        batch_size: Samples per batch.
        shuffle: Whether to shuffle (ignored when balance_classes=True).
        num_workers: Parallel data loading workers.
        pin_memory: Pin memory for faster GPU transfer.
        balance_classes: Use WeightedRandomSampler to oversample minority class.

    Returns:
        Configured DataLoader.
    """
    sampler = None
    _shuffle = shuffle

    if balance_classes:
        weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        _shuffle = False  # sampler and shuffle are mutually exclusive

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def get_dataloaders(config) -> dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders from config.

    Returns:
        dict with keys "train", "val", "test".
    """
    cls_config = config.classification
    hw_config = config.hardware

    img_size = cls_config.input.img_size
    batch_size = cls_config.training.batch_size
    num_workers = hw_config.num_workers
    pin_memory = hw_config.pin_memory
    class_names = config.classes.names

    loaders = {}

    # Training set
    train_dataset = ArecanutDataset(
        root_dir=config.paths.train_dir,
        transform=get_train_transforms(img_size),
        class_names=class_names,
    )
    loaders["train"] = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Using sampler
        num_workers=num_workers,
        pin_memory=pin_memory,
        balance_classes=True,
    )
    logger.info(f"Train:\n{train_dataset.summary()}")

    # Validation set
    val_dataset = ArecanutDataset(
        root_dir=config.paths.val_dir,
        transform=get_val_transforms(img_size),
        class_names=class_names,
    )
    loaders["val"] = build_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    logger.info(f"Val:\n{val_dataset.summary()}")

    # Test set
    test_dataset = ArecanutDataset(
        root_dir=config.paths.test_dir,
        transform=get_val_transforms(img_size),
        class_names=class_names,
    )
    loaders["test"] = build_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    logger.info(f"Test:\n{test_dataset.summary()}")

    return loaders


# ---------------------------------------------------------------------------
# Dataset Statistics
# ---------------------------------------------------------------------------

def compute_dataset_stats(dataset: ArecanutDataset, num_workers: int = 0) -> dict:
    """
    Compute per-channel mean and std of the dataset (used for custom normalization).

    Args:
        dataset: Dataset with ToTensor() transform (no normalization).
        num_workers: DataLoader workers.

    Returns:
        dict with 'mean' and 'std' lists.
    """
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
    )

    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    total_pixels = 0

    for images, _ in loader:
        B, C, H, W = images.shape
        total_pixels += B * H * W
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sq_sum += (images ** 2).sum(dim=[0, 2, 3])

    mean = (channel_sum / total_pixels).tolist()
    std = ((channel_sq_sum / total_pixels - torch.tensor(mean) ** 2) ** 0.5).tolist()

    return {"mean": [round(m, 4) for m in mean], "std": [round(s, 4) for s in std]}


# ---------------------------------------------------------------------------
# Dataset Preparation Helpers
# ---------------------------------------------------------------------------

def split_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Split images from source_dir (containing class subdirs) into
    train/val/test splits under output_dir.

    Args:
        source_dir: Directory with class subdirs (ripe/, unripe/).
        output_dir: Root output directory for split dataset.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed for reproducibility.

    Returns:
        Split count summary dict.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    rng = np.random.default_rng(seed)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    summary = {}
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        cls = class_dir.name
        images = sorted([
            p for p in class_dir.iterdir()
            if p.suffix.lower() in EXTENSIONS
        ])
        rng.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        summary[cls] = {}
        for split_name, split_images in splits.items():
            dest = output_dir / split_name / cls
            dest.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy2(img_path, dest / img_path.name)
            summary[cls][split_name] = len(split_images)
            logger.info(f"  {cls}/{split_name}: {len(split_images)} images → {dest}")

    logger.info(f"Dataset split complete. Summary:\n{json.dumps(summary, indent=2)}")
    return summary


def validate_dataset_structure(dataset_root: str | Path) -> bool:
    """
    Validate that the dataset has the expected folder structure and
    contains images. Returns True if valid.
    """
    dataset_root = Path(dataset_root)
    required_splits = ["train", "val", "test"]
    required_classes = ["ripe", "unripe"]
    valid = True

    for split in required_splits:
        for cls in required_classes:
            path = dataset_root / split / cls
            if not path.exists():
                logger.error(f"Missing: {path}")
                valid = False
            else:
                imgs = list(path.glob("*.[jJpP][pPnN][gGeE]*"))
                if not imgs:
                    logger.warning(f"Empty directory: {path}")
                else:
                    logger.info(f"  {split}/{cls}: {len(imgs)} images ✓")

    return valid
