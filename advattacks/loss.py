from __future__ import annotations

import typing as t

import torch

if t.TYPE_CHECKING:
    from advattacks.wrapper import Wrapper


class TargetLoss:
    """Compute teacher-forcing cross-entropy loss averaged across models
    and prefixes.

    Implements:
        L_target(x) = 1/|M| * sum_M [ 1/K * sum_k [ -sum_t log P_M(y_t^(k) | T, x, y_{<t}^(k)) ] ]

    Args:
        wrappers: List of model wrappers.
        target_token_ids: List of tokenized target prefixes, shape
            [(seq_len_k,), ... ].
    """

    def __init__(self, wrappers: list[Wrapper], target_token_ids: list[torch.Tensor]):
        self.wrappers = wrappers
        self.target_token_ids = target_token_ids

    def __call__(self, image: torch.Tensor, text: str) -> torch.Tensor:
        """Compute target loss for given image and text.

        Args:
            image: Image tensor of shape (C, H, W) in [0, 1] range.
            text: Text prompt.

        Returns:
            Scalar target loss.
        """
        total_loss = torch.tensor(0.0, device=image.device)
        num_models = len(self.wrappers)
        num_prefixes = len(self.target_token_ids)

        for wrapper in self.wrappers:
            model_loss = 0.0

            for target_ids in self.target_token_ids:
                # Compute teacher-forcing cross-entropy
                loss = wrapper.compute_loss(image, text, target_ids)
                model_loss += loss

            # Average over prefixes
            total_loss += model_loss / num_prefixes

        # Average over models
        return total_loss / num_models


class CovertLoss:
    """Compute L-infinity norm penalty for image perturbation.

    Implements:
        L_inf = ||x - x_0||_inf

    Args:
        original_image: Original clean image tensor.
    """

    def __init__(self, original_image: torch.Tensor):
        self.original_image = original_image

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Compute L-infinity norm between current and original image.

        Args:
            image: Current image tensor.

        Returns:
            Scalar L-infinity norm.
        """
        return torch.norm(image - self.original_image, p=float("inf"))


class CompositeLoss:
    """Composite loss combining target and covert losses.

    Implements:
        L(x) = lambda_target * L_target(x) + lambda_inf * L_inf(x)

    Args:
        target_loss: Target loss function.
        covert_loss: Covert loss function.
        lambda_target: Weight for target loss.
        lambda_inf: Weight for covert loss.
    """

    def __init__(
        self,
        target_loss: TargetLoss,
        covert_loss: CovertLoss,
        lambda_target: float = 1.0,
        lambda_inf: float = 0.01,
    ):
        self.target_loss = target_loss
        self.covert_loss = covert_loss
        self.lambda_target = lambda_target
        self.lambda_inf = lambda_inf

    def __call__(self, image: torch.Tensor, text: str) -> torch.Tensor:
        """Compute composite loss.

        Args:
            image: Image tensor.
            text: Text prompt.

        Returns:
            Scalar composite loss.
        """
        target = self.target_loss(image, text)
        covert = self.covert_loss(image)

        return self.lambda_target * target + self.lambda_inf * covert
