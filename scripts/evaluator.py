from __future__ import annotations

import typing as t

import torch
import torch.nn.functional as F  # noqa: N812


class ComparisonMetrics(t.TypedDict):
    question_id: str
    scenario: str
    linf_norm: float
    l2_norm: float
    psnr: float
    original_response_length: int
    adversarial_response_length: int


def compute_psnr(original: torch.Tensor, adversarial: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        original: Original image tensor.
        adversarial: Adversarial image tensor.

    Returns:
        PSNR value in dB.
    """
    mse = F.mse_loss(original, adversarial)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def compute_metrics(
    question_id: str,
    scenario: str,
    original_image: torch.Tensor,
    adversarial_image: torch.Tensor,
    original_response: str,
    adversarial_response: str,
) -> ComparisonMetrics:
    """Compute comparison metrics.

    Args:
        question_id: Question identifier.
        scenario: Scenario name.
        original_image: Original image.
        adversarial_image: Adversarial image.
        original_response: Model response on original image.
        adversarial_response: Model response on adversarial image.

    Returns:
        Comparison metrics.
    """
    linf_norm = torch.norm(adversarial_image - original_image, p=float("inf")).item()
    l2_norm = torch.norm(adversarial_image - original_image, p=2).item()
    psnr = compute_psnr(original_image, adversarial_image)

    return {
        "question_id": question_id,
        "scenario": scenario,
        "linf_norm": linf_norm,
        "l2_norm": l2_norm,
        "psnr": psnr,
        "original_response_length": len(original_response),
        "adversarial_response_length": len(adversarial_response),
    }
