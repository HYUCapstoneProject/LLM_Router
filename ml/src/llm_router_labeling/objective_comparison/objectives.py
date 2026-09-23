from __future__ import annotations

from typing import Any, Sequence


METHODS = ("ordinal", "pointwise", "pairwise", "regret")


def require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Routing-objective training requires PyTorch; install the model-training extras"
        ) from error
    return torch


def objective_loss(
    method: str,
    logits: Any,
    *,
    difficulty_ids: Any,
    success_targets: Any,
    quality_targets: Any,
    decision_costs: Any,
    sample_weights: Any,
    regret_lambdas: Sequence[float],
    pairwise_tie_weight: float,
) -> Any:
    """Compute one of four losses over identical three-action logits."""
    if method not in METHODS:
        raise ValueError(f"Unsupported objective method: {method}")
    torch = require_torch()
    functional = torch.nn.functional
    weights = sample_weights.to(logits.dtype)
    if method == "ordinal":
        values = functional.cross_entropy(logits, difficulty_ids, reduction="none")
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    if method == "pointwise":
        values = functional.binary_cross_entropy_with_logits(
            logits,
            success_targets.to(logits.dtype),
            reduction="none",
        ).mean(dim=1)
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    if method == "pairwise":
        pair_losses = []
        pair_weights = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            quality_delta = quality_targets[:, left] - quality_targets[:, right]
            target = torch.where(
                quality_delta > 0,
                torch.ones_like(quality_delta),
                torch.where(
                    quality_delta < 0,
                    torch.zeros_like(quality_delta),
                    torch.full_like(quality_delta, 0.5),
                ),
            )
            importance = quality_delta.abs()
            importance = torch.where(
                importance > 0,
                importance,
                torch.full_like(importance, float(pairwise_tie_weight)),
            )
            pair_losses.append(
                functional.binary_cross_entropy_with_logits(
                    logits[:, left] - logits[:, right], target, reduction="none"
                )
            )
            pair_weights.append(importance)
        values = torch.stack(pair_losses, dim=1)
        importance = torch.stack(pair_weights, dim=1)
        values = (values * importance).sum(dim=1) / importance.sum(dim=1).clamp_min(1e-8)
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    lambda_tensor = torch.tensor(
        list(regret_lambdas), device=logits.device, dtype=logits.dtype
    ).view(1, -1, 1)
    utilities = quality_targets.to(logits.dtype).unsqueeze(1) - (
        lambda_tensor * decision_costs.to(logits.dtype).unsqueeze(1)
    )
    best_utility, best_actions = utilities.max(dim=2)
    second_utility = utilities.topk(k=2, dim=2).values[:, :, 1]
    action_gap = (best_utility - second_utility).clamp_min(1e-4)
    adjusted_logits = logits.unsqueeze(1) - (
        lambda_tensor * decision_costs.to(logits.dtype).unsqueeze(1)
    )
    flat_loss = functional.cross_entropy(
        adjusted_logits.reshape(-1, adjusted_logits.shape[-1]),
        best_actions.reshape(-1),
        reduction="none",
    ).reshape(logits.shape[0], -1)
    values = (flat_loss * action_gap).mean(dim=1)
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)
