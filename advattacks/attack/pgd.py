from __future__ import annotations

import typing as t

import torch

from advattacks import utils
from advattacks.attack import Attack
from advattacks.loss import CompositeLoss
from advattacks.loss import CovertLoss
from advattacks.loss import TargetLoss
from advattacks.prefixes import DEFAULT_PREFIXES
from advattacks.prefixes import tokenize_prefixes

if t.TYPE_CHECKING:
    from advattacks.wrapper import Wrapper


class PGD(Attack):
    """Projected Gradient Descent attack with round-robin model loading.

    Implements the iterative attack:
        1. g_t = ∇_x L(x_t)
        2. x̃_{t+1} = x_t - α * sign(g_t)
        3. x_{t+1} = clip(x̃_{t+1}, x_0 - ε, x_0 + ε) ∩ [0, 1]

    Args:
        wrappers: Sequence of model wrappers to attack.
        epsilon: Maximum L-infinity perturbation (default: 32/255).
        alpha: Step size for each PGD iteration (default: 2/255).
        num_rounds: Number of complete rounds through all models.
        steps_per_model: Number of PGD steps per model per round.
        lambda_target: Weight for target loss.
        lambda_inf: Weight for covert loss.
        prefixes: Sequence of target prefix strings for jailbreak.
    """

    def __init__(
        self,
        wrappers: t.Sequence[Wrapper],
        epsilon: float = 32 / 255,
        alpha: float = 2 / 255,
        num_rounds: int = 4,
        steps_per_model: int = 5,
        lambda_target: float = 1.0,
        lambda_inf: float = 0.01,
        prefixes: t.Sequence[str] | None = None,
    ):
        super().__init__(wrappers, epsilon, prefixes)
        self.alpha = alpha
        self.num_rounds = num_rounds
        self.steps_per_model = steps_per_model
        self.lambda_target = lambda_target
        self.lambda_inf = lambda_inf

        if self.prefixes is None:
            self.prefixes = DEFAULT_PREFIXES

        # Cache for tokenized prefixes per wrapper
        self._tokenized_cache: dict[Wrapper, list[torch.Tensor]] = {}

    def _get_tokenized_prefixes(self, wrapper: Wrapper) -> list[torch.Tensor]:
        """Get or compute tokenized prefixes for a wrapper.

        Args:
            wrapper: Model wrapper.

        Returns:
            List of tokenized prefix tensors.
        """
        if wrapper not in self._tokenized_cache:
            self._tokenized_cache[wrapper] = tokenize_prefixes(self.prefixes, wrapper)
        return self._tokenized_cache[wrapper]

    def _pgd_step(
        self,
        image: torch.Tensor,
        original_image: torch.Tensor,
        text: str,
        wrapper: Wrapper,
    ) -> torch.Tensor:
        """Perform a single PGD update step.

        Args:
            image: Current perturbed image.
            original_image: Original clean image.
            text: Text prompt.
            wrapper: Model wrapper to compute gradient against.

        Returns:
            Updated perturbed image.
        """
        # Get tokenized prefixes for this wrapper
        target_token_ids = self._get_tokenized_prefixes(wrapper)

        # Construct loss functions
        target_loss = TargetLoss([wrapper], target_token_ids)
        covert_loss = CovertLoss(original_image)
        composite_loss = CompositeLoss(
            target_loss,
            covert_loss,
            self.lambda_target,
            self.lambda_inf,
        )

        # Compute gradient
        image_adv = image.clone().detach().requires_grad_(True)
        loss = composite_loss(image_adv, text)
        grad = torch.autograd.grad(loss, image_adv)[0]

        # PGD update: x - alpha * sign(grad)
        normalized_grad = utils.normalize_gradient(grad)
        image_updated = image_adv - self.alpha * normalized_grad

        # Project back to epsilon-ball and [0, 1]
        image_projected = utils.project_linf(
            image_updated,
            original_image,
            self.epsilon,
        )

        return image_projected.detach()

    def attack(self, image: torch.Tensor, text: str, verbose: bool = True) -> torch.Tensor:
        """Run PGD attack with round-robin model loading.

        Args:
            image: Original clean image tensor of shape (C, H, W) in [0, 1] range.
            text: Text prompt.
            verbose: Whether to print progress.

        Returns:
            Adversarial image tensor.
        """
        original_image = image.clone().detach()
        x = original_image.clone()

        total_steps = self.num_rounds * len(self.wrappers) * self.steps_per_model
        current_step = 0

        for round_idx in range(self.num_rounds):
            if verbose:
                print(f"\n=== Round {round_idx + 1}/{self.num_rounds} ===")

            for wrapper_idx, wrapper in enumerate(self.wrappers):
                # Load model
                if verbose:
                    print(f"  Loading model {wrapper_idx + 1}/{len(self.wrappers)}...")
                wrapper.load()

                # Perform PGD steps for this model
                for _ in range(self.steps_per_model):
                    x = self._pgd_step(x, original_image, text, wrapper)
                    current_step += 1

                    if verbose:
                        linf_norm = torch.norm(x - original_image, p=float("inf")).item()
                        print(f"    Step {current_step}/{total_steps} | L∞: {linf_norm:.6f}")

                # Unload model and free memory
                if verbose:
                    print(f"  Unloading model {wrapper_idx + 1}...")
                wrapper.unload()

        if verbose:
            final_linf = torch.norm(x - original_image, p=float("inf")).item()
            print("\n=== Attack Complete ===")
            print(f"Final L∞ norm: {final_linf:. 6f}")
            print(f"Epsilon constraint: {self.epsilon:.6f}")
            print(f"Constraint satisfied: {final_linf <= self.epsilon}")

        return x
