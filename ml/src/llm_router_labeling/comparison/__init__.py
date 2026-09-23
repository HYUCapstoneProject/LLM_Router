"""Shared training and evaluation contracts for router model comparison."""

from .evaluation import LABELS, evaluate_classification_predictions
from .tfidf_baseline import run_tfidf_baseline

__all__ = ["LABELS", "evaluate_classification_predictions", "run_tfidf_baseline"]
