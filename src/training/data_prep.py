"""Data preparation helpers for satellite model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json
import random
import shutil

import numpy as np

from geospatial.hyperspectral import HyperspectralProcessor

try:
    import rasterio  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    rasterio = None


EUROSAT_TO_LANDUSE: Dict[str, str] = {
    "AnnualCrop": "cropland",
    "Forest": "forest",
    "HerbaceousVegetation": "grassland",
    "Highway": "urban",
    "Industrial": "urban",
    "Pasture": "grassland",
    "PermanentCrop": "cropland",
    "Residential": "urban",
    "River": "water",
    "SeaLake": "water",
}

WORLD_COVER_MAP: Dict[int, str] = {
    10: "forest",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "urban",
    60: "barren",
    70: "snow",
    80: "water",
    90: "wetland",
    95: "wetland",
    100: "mixed",
}

DYNAMIC_WORLD_MAP: Dict[int, str] = {
    0: "water",
    1: "forest",
    2: "grassland",
    3: "wetland",
    4: "cropland",
    5: "shrubland",
    6: "urban",
    7: "barren",
    8: "snow",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass
class EuroSATPrepConfig:
    source_dir: str
    output_dir: str
    val_ratio: float = 0.2
    seed: int = 42


@dataclass
class EmitPseudoLabelConfig:
    emit_cube_path: str
    label_raster_path: str
    output_dir: str
    label_source: str = "worldcover"
    chip_size: int = 64
    stride: int = 64
    val_ratio: float = 0.2
    min_valid_fraction: float = 0.8
    min_majority_fraction: float = 0.7
    max_chips_per_class: int = 2000
    seed: int = 42


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def prepare_eurosat(cfg: EuroSATPrepConfig) -> Dict[str, int]:
    source = Path(cfg.source_dir)
    output = Path(cfg.output_dir)
    if not source.exists():
        raise FileNotFoundError(f"EuroSAT source directory not found: {source}")

    rng = random.Random(cfg.seed)
    counts: Dict[str, int] = {}

    for eurosat_class, target_class in EUROSAT_TO_LANDUSE.items():
        class_dir = source / eurosat_class
        if not class_dir.exists():
            continue
        files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        if not files:
            continue
        rng.shuffle(files)

        val_size = max(1, int(len(files) * cfg.val_ratio))
        val_files = files[:val_size]
        train_files = files[val_size:]
        if not train_files:
            train_files = val_files
            val_files = []

        for src in train_files:
            dst = output / "train" / target_class / src.name
            _safe_copy(src, dst)

        for src in val_files:
            dst = output / "val" / target_class / src.name
            _safe_copy(src, dst)

        counts[target_class] = counts.get(target_class, 0) + len(files)

    if not counts:
        raise ValueError("No EuroSAT class folders were found or contained supported image files.")

    summary_path = output / "eurosat_prep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"class_counts": counts}, indent=2))
    return counts


def _choose_label_map(label_source: str) -> Dict[int, str]:
    source = label_source.lower().strip()
    if source == "worldcover":
        return WORLD_COVER_MAP
    if source == "dynamicworld":
        return DYNAMIC_WORLD_MAP
    raise ValueError("label_source must be one of: worldcover, dynamicworld")


def _split_train_val(items: List[Tuple[np.ndarray, str]], val_ratio: float, seed: int) -> Tuple[List[Tuple[np.ndarray, str]], List[Tuple[np.ndarray, str]]]:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val = shuffled[:val_size]
    train = shuffled[val_size:]
    if not train and val:
        train, val = val, []
    return train, val


def prepare_emit_pseudolabels(cfg: EmitPseudoLabelConfig) -> Dict[str, int]:
    if rasterio is None:
        raise ImportError("rasterio is required for label raster reading. Install rasterio>=1.3.0.")

    label_map = _choose_label_map(cfg.label_source)
    cube = HyperspectralProcessor.load_cube(cfg.emit_cube_path)

    with rasterio.open(cfg.label_raster_path) as src:
        labels = src.read(1)

    if labels.shape[0] != cube.shape[0] or labels.shape[1] != cube.shape[1]:
        raise ValueError(
            f"Label raster shape {labels.shape} does not match EMIT cube spatial shape {(cube.shape[0], cube.shape[1])}."
        )

    candidates: List[Tuple[np.ndarray, str]] = []
    class_counts: Dict[str, int] = {}
    height, width, _bands = cube.shape

    for row in range(0, max(1, height - cfg.chip_size + 1), cfg.stride):
        for col in range(0, max(1, width - cfg.chip_size + 1), cfg.stride):
            chip = cube[row:row + cfg.chip_size, col:col + cfg.chip_size, :]
            label_chip = labels[row:row + cfg.chip_size, col:col + cfg.chip_size]
            if chip.shape[0] != cfg.chip_size or chip.shape[1] != cfg.chip_size:
                continue

            valid = label_chip[label_chip > 0]
            valid_fraction = valid.size / float(label_chip.size)
            if valid_fraction < cfg.min_valid_fraction:
                continue

            values, counts = np.unique(valid.astype(int), return_counts=True)
            majority_idx = int(np.argmax(counts))
            raw_label = int(values[majority_idx])
            majority_fraction = float(counts[majority_idx]) / float(valid.size)
            if majority_fraction < cfg.min_majority_fraction:
                continue
            if raw_label not in label_map:
                continue

            target_class = label_map[raw_label]
            if class_counts.get(target_class, 0) >= cfg.max_chips_per_class:
                continue

            candidates.append((chip.astype(np.float32), target_class))
            class_counts[target_class] = class_counts.get(target_class, 0) + 1

    if not candidates:
        raise ValueError("No pseudolabeled chips were generated. Relax thresholds or verify label raster alignment.")

    train_items, val_items = _split_train_val(candidates, cfg.val_ratio, cfg.seed)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_idx: Dict[str, int] = {}
    val_idx: Dict[str, int] = {}

    for chip, cls in train_items:
        idx = train_idx.get(cls, 0)
        path = output / "train" / cls / f"{cls}_{idx:06d}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, chip)
        train_idx[cls] = idx + 1

    for chip, cls in val_items:
        idx = val_idx.get(cls, 0)
        path = output / "val" / cls / f"{cls}_{idx:06d}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, chip)
        val_idx[cls] = idx + 1

    summary = {
        "label_source": cfg.label_source,
        "chip_size": cfg.chip_size,
        "stride": cfg.stride,
        "train_counts": train_idx,
        "val_counts": val_idx,
        "total_candidates": len(candidates),
    }
    (output / "emit_pseudolabel_summary.json").write_text(json.dumps(summary, indent=2))
    return {k: int(v) for k, v in class_counts.items()}

