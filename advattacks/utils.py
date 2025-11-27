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
    # Clip to epsilon ball
    x_proj = torch.clamp(x, x_original - epsilon, x_original + epsilon)

    # Clip to valid pixel range
    return torch.clamp(x_proj, 0.0, 1.0)


def normalize_gradient(grad: torch.Tensor) -> torch.Tensor:
    """Normalize gradient to unit L-infinity norm (i.e., sign of
    gradient).

    Args:
        grad: Gradient tensor.

    Returns:
        Normalized gradient (sign of input).
    """
    return torch.sign(grad)
