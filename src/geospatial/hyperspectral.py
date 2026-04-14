"""Utility helpers for loading and summarizing hyperspectral cubes."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class HyperspectralProcessor:
    """Utilities for hyperspectral image ingestion and analysis."""

    @staticmethod
    def load_cube(path: str) -> np.ndarray:
        """Load a hyperspectral cube from .npy, .npz, .nc, or .txt/.csv formats."""
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Hyperspectral cube not found: {resolved}")

        if resolved.suffix == ".npy":
            cube = np.load(resolved)
        elif resolved.suffix == ".npz":
            data = np.load(resolved)
            cube = next(iter(data.values()))
        elif resolved.suffix == ".nc":
            try:
                import xarray as xr  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("xarray is required to load NetCDF hyperspectral cubes.") from exc

            ds = xr.open_dataset(resolved)
            preferred_vars = ("reflectance", "radiance")
            variable_name = next((name for name in preferred_vars if name in ds.data_vars), None)
            if not variable_name:
                if len(ds.data_vars) != 1:
                    raise ValueError(
                        "Could not infer hyperspectral variable from NetCDF. Expected 'reflectance' or 'radiance'."
                    )
                variable_name = next(iter(ds.data_vars))
            cube = ds[variable_name].values
        elif resolved.suffix in {".txt", ".csv"}:
            cube = np.loadtxt(resolved, delimiter=",")
        else:
            raise ValueError(
                "Unsupported hyperspectral file format. Use .npy, .npz, .nc, or a CSV-like text file."
            )

        if cube.ndim != 3:
            raise ValueError("Hyperspectral cube must be a 3D array shaped (height, width, bands).")
        return cube.astype(np.float32)

    @staticmethod
    def select_bands(cube: np.ndarray, bands: Sequence[int]) -> np.ndarray:
        """Return an RGB composite from specified band indexes."""
        if len(bands) != 3:
            raise ValueError("Exactly three bands are required to build an RGB composite.")
        height, width, num_bands = cube.shape
        indices = []
        for band in bands:
            if band < 0 or band >= num_bands:
                raise ValueError(f"Band index {band} is outside the cube range (0-{num_bands - 1}).")
            indices.append(band)
        rgb = cube[:, :, indices]
        return HyperspectralProcessor.normalize(rgb)

    @staticmethod
    def pca_to_rgb(cube: np.ndarray, n_components: int = 3) -> np.ndarray:
        """Compress the spectral cube to RGB using PCA derived from SVD."""
        height, width, bands = cube.shape
        reshaped = cube.reshape(-1, bands)
        reshaped = reshaped - reshaped.mean(axis=0, keepdims=True)
        _u, _s, vh = np.linalg.svd(reshaped, full_matrices=False)
        comps = np.dot(reshaped, vh[:n_components].T)
        rgb = comps.reshape(height, width, n_components)
        return HyperspectralProcessor.normalize(rgb)

    @staticmethod
    def normalize(rgb: np.ndarray) -> np.ndarray:
        rgb_min = rgb.min()
        rgb_max = rgb.max()
        if rgb_max - rgb_min < 1e-8:
            return np.zeros_like(rgb)
        return (rgb - rgb_min) / (rgb_max - rgb_min)

    @staticmethod
    def compute_signature(
        cube: np.ndarray,
        sample_fraction: float = 0.02,
        sample_coordinates: Optional[Iterable[Tuple[int, int]]] = None,
    ) -> Dict[str, List[float]]:
        """Summarize the spectral signature by averaging sampled pixels."""
        height, width, bands = cube.shape
        pixels = cube.reshape(-1, bands)

        if sample_coordinates:
            idx = []
            for row, col in sample_coordinates:
                if 0 <= row < height and 0 <= col < width:
                    idx.append(row * width + col)
            if not idx:
                raise ValueError("Provided sample coordinates are outside cube bounds.")
            sampled = pixels[idx]
        else:
            total_pixels = pixels.shape[0]
            sample_size = max(1, int(total_pixels * sample_fraction))
            rng = np.random.default_rng(42)
            selected = rng.choice(total_pixels, size=sample_size, replace=False)
            sampled = pixels[selected]

        signature = sampled.mean(axis=0)
        return {
            "bands": bands,
            "mean_spectrum": signature.astype(float).tolist(),
        }


__all__ = ["HyperspectralProcessor"]
