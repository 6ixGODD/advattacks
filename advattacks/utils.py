from __future__ import annotations

import torch


def project_linf(x: torch.Tensor, x_original: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Project perturbation into L-infinity epsilon-ball and [0, 1]
    range.

    Implements:
        1. x_proj = clip(x, x_0 - epsilon, x_0 + epsilon)
        2. x_proj = clip(x_proj, 0, 1)

    Args:
        x: Perturbed image tensor.
        x_original: Original clean image tensor.
        epsilon: Maximum L-infinity perturbation.

    Returns:
        Projected image tensor.
    """
    # Compute bounds
    lower_bound = torch.max(x_original - epsilon, torch.zeros_like(x_original))
    upper_bound = torch.min(x_original + epsilon, torch.ones_like(x_original))

    # Project into bounds
    return torch.clamp(x, lower_bound, upper_bound)


def normalize_gradient(grad: torch.Tensor) -> torch.Tensor:
    """Normalize gradient to unit L-infinity norm (i.e., sign of
    gradient).

    Args:
        grad: Gradient tensor.

    Returns:
        Normalized gradient (sign of input).
    """
    return torch.sign(grad)
