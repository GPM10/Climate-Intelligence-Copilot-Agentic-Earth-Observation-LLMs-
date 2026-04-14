"""Training utilities for fine-tuning the satellite ResNet classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

from geospatial.hyperspectral import HyperspectralProcessor


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
HYPERSPECTRAL_SUFFIXES = {".npy", ".npz", ".nc", ".txt", ".csv"}


@dataclass
class TrainConfig:
    data_dir: str
    output_path: str
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 1e-4
    val_ratio: float = 0.2
    seed: int = 42
    pretrained: bool = True
    freeze_backbone: bool = False
    device: str = "cpu"
    num_workers: int = 0
    rgb_mode: str = "band_selection"
    hyperspectral_bands: Sequence[int] = (10, 30, 50)


class SatelliteTrainingDataset(Dataset):
    """Dataset that supports both RGB imagery and hyperspectral cubes."""

    def __init__(self, class_dirs: List[Path], class_names: List[str], rgb_mode: str, hyperspectral_bands: Sequence[int]):
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.samples: List[Tuple[Path, int]] = []
        self.rgb_mode = rgb_mode
        self.hyperspectral_bands = hyperspectral_bands

        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            for path in sorted(class_dir.rglob("*")):
                if not path.is_file():
                    continue
                suffix = path.suffix.lower()
                if suffix in IMAGE_SUFFIXES or suffix in HYPERSPECTRAL_SUFFIXES:
                    self.samples.append((path, class_idx))

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = self._load_image(path)
        image = image.float()

        if image.max() > 1:
            image = image / 255.0

        # Shape to CxHxW
        if image.ndim == 3 and image.shape[-1] in (1, 3):
            image = image.permute(2, 0, 1)
        elif image.ndim == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)

        image = image.unsqueeze(0)
        image = torch.nn.functional.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        image = image.squeeze(0)
        image = self.normalize(image)

        return image, label

    def _load_image(self, path: Path) -> torch.Tensor:
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            return torch.tensor(np.array(Image.open(path).convert("RGB")), dtype=torch.float32)

        cube = HyperspectralProcessor.load_cube(str(path))
        if self.rgb_mode == "pca":
            rgb = HyperspectralProcessor.pca_to_rgb(cube)
        else:
            rgb = HyperspectralProcessor.select_bands(cube, self.hyperspectral_bands)
        return torch.tensor(rgb, dtype=torch.float32)


def _build_datasets(cfg: TrainConfig) -> Tuple[Dataset, Dataset, List[str]]:
    root = Path(cfg.data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Training data directory not found: {root}")

    train_root = root / "train"
    val_root = root / "val"
    has_split = train_root.exists() and val_root.exists()

    if has_split:
        train_class_dirs = sorted([p for p in train_root.iterdir() if p.is_dir()])
        val_class_dirs = sorted([p for p in val_root.iterdir() if p.is_dir()])
        class_names = sorted({p.name for p in train_class_dirs + val_class_dirs})

        train_ds = SatelliteTrainingDataset(train_class_dirs, class_names, cfg.rgb_mode, cfg.hyperspectral_bands)
        val_ds = SatelliteTrainingDataset(val_class_dirs, class_names, cfg.rgb_mode, cfg.hyperspectral_bands)
        return train_ds, val_ds, class_names

    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if len(class_dirs) < 2:
        raise ValueError(
            "Training directory must contain class subfolders (at least two). "
            "Example: data/train/satellite/<class_name>/*.npy"
        )

    class_names = [p.name for p in class_dirs]
    full_ds = SatelliteTrainingDataset(class_dirs, class_names, cfg.rgb_mode, cfg.hyperspectral_bands)
    if len(full_ds) < 2:
        raise ValueError("Not enough samples for train/validation split.")

    val_size = max(1, int(len(full_ds) * cfg.val_ratio))
    train_size = len(full_ds) - val_size
    if train_size <= 0:
        raise ValueError("Validation ratio too high for dataset size.")

    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=generator)
    return train_ds, val_ds, class_names


def train_satellite_model(cfg: TrainConfig) -> Dict[str, str]:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_ds, val_ds, class_names = _build_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    weights = models.ResNet50_Weights.IMAGENET1K_V1 if cfg.pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    if cfg.freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_path = output_path.with_name(f"{output_path.stem}.best{output_path.suffix}")

    for _epoch in range(cfg.epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                val_loss_sum += float(loss.item()) * labels.size(0)
                val_count += labels.size(0)

        val_loss = val_loss_sum / max(1, val_count)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "num_classes": len(class_names),
                    "rgb_mode": cfg.rgb_mode,
                    "hyperspectral_bands": list(cfg.hyperspectral_bands),
                },
                best_path,
            )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "num_classes": len(class_names),
            "rgb_mode": cfg.rgb_mode,
            "hyperspectral_bands": list(cfg.hyperspectral_bands),
        },
        output_path,
    )

    return {
        "checkpoint": str(output_path),
        "best_checkpoint": str(best_path),
        "num_classes": str(len(class_names)),
    }

