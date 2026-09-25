"""Fresh-initialization task/domain generalization experiment support."""

from .data import (
    GeneralizationDataError,
    build_generalization_dataset,
    canonical_config_sha256,
    load_role,
    load_training_view,
    verify_generalization_dataset,
)

__all__ = [
    "GeneralizationDataError",
    "build_generalization_dataset",
    "canonical_config_sha256",
    "load_role",
    "load_training_view",
    "verify_generalization_dataset",
]
