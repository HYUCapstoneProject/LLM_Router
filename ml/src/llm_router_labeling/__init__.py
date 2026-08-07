"""LLM router data, training, and evaluation pipelines.

RouterBench helpers remain available from this top-level module, but they are
loaded lazily so the ModernBERT training image does not need to include the CPU
labeling implementation.
"""

from typing import Any

__all__ = [
    "calibrate_model_tiers",
    "canonicalize_sources",
    "detect_model_columns",
    "label_questions",
    "stratified_smoke_sample",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .training import routerbench_labeling

    return getattr(routerbench_labeling, name)
