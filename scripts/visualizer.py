from __future__ import annotations

import pathlib
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_comparison(
    original: torch.Tensor,
    adversarial: torch.Tensor,
    perturbation: torch.Tensor,
    question_id: str,
    scenario: str,
    output_path: pathlib.Path,
) -> None:
    """Create a comparison visualization.

    Args:
        original: Original image tensor (C, H, W).
        adversarial: Adversarial image tensor (C, H, W).
        perturbation: Perturbation tensor (C, H, W).
        question_id: Question identifier.
        scenario: Scenario name.
        output_path: Path to save the visualization.
    """
    _fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Convert tensors to numpy arrays
    original_np = original.permute(1, 2, 0).cpu().numpy()
    adversarial_np = adversarial.permute(1, 2, 0).cpu().numpy()
    perturbation_np = perturbation.permute(1, 2, 0).cpu().numpy()

    # Normalize perturbation for visualization
    pert_min = perturbation_np.min()
    pert_max = perturbation_np.max()
    if pert_max - pert_min > 0:
        perturbation_vis = (perturbation_np - pert_min) / (pert_max - pert_min)
    else:
        perturbation_vis = perturbation_np

    # Plot original
    axes[0].imshow(np.clip(original_np, 0, 1))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Plot adversarial
    axes[1].imshow(np.clip(adversarial_np, 0, 1))
    linf = torch.norm(perturbation, p=float("inf")).item()
    axes[1].set_title(f"Adversarial Image\n(L∞={linf:.4f})")
    axes[1].axis("off")

    # Plot perturbation
    axes[2].imshow(perturbation_vis)
    axes[2].set_title("Perturbation\n(normalized)")
    axes[2].axis("off")

    plt.suptitle(f"{scenario} - Question {question_id}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_summary(
    metrics_list: list[t.Mapping[str, object]],
    output_path: pathlib.Path,
) -> None:
    """Plot summary of metrics across all samples.

    Args:
        metrics_list: List of metrics dictionaries.
        output_path: Path to save the plot.
    """
    _fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # L-infinity norms
    linf_norms = [m["linf_norm"] for m in metrics_list]
    axes[0, 0].hist(linf_norms, bins=20, edgecolor="black")
    axes[0, 0].set_title("L∞ Norm Distribution")
    axes[0, 0].set_xlabel("L∞ Norm")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].axvline(32 / 255, color="r", linestyle="--", label="ε=32/255")
    axes[0, 0].legend()

    # L2 norms
    l2_norms = [m["l2_norm"] for m in metrics_list]
    axes[0, 1].hist(l2_norms, bins=20, edgecolor="black")
    axes[0, 1].set_title("L2 Norm Distribution")
    axes[0, 1].set_xlabel("L2 Norm")
    axes[0, 1].set_ylabel("Count")

    # PSNR
    psnr_values = [m["psnr"] for m in metrics_list if m["psnr"] != float("inf")]
    axes[1, 0].hist(psnr_values, bins=20, edgecolor="black")
    axes[1, 0].set_title("PSNR Distribution")
    axes[1, 0].set_xlabel("PSNR (dB)")
    axes[1, 0].set_ylabel("Count")

    # Response length comparison
    orig_lengths = [m["original_response_length"] for m in metrics_list]
    adv_lengths = [m["adversarial_response_length"] for m in metrics_list]
    x = np.arange(len(metrics_list))
    width = 0.35
    axes[1, 1].bar(x - width / 2, orig_lengths, width, label="Original", alpha=0.8)
    axes[1, 1].bar(x + width / 2, adv_lengths, width, label="Adversarial", alpha=0.8)
    axes[1, 1].set_title("Response Length Comparison")
    axes[1, 1].set_xlabel("Sample Index")
    axes[1, 1].set_ylabel("Response Length")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
