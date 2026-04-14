"""Training utilities for Climate Intelligence Copilot."""

from .satellite import train_satellite_model
from .data_prep import prepare_eurosat, prepare_emit_pseudolabels

__all__ = ["train_satellite_model", "prepare_eurosat", "prepare_emit_pseudolabels"]
