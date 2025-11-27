from __future__ import annotations

import typing as t

import torch

from advattacks import utils
from advattacks.attack import Attack
from advattacks.prefixes import DEFAULT_PREFIXES

if t.TYPE_CHECKING:
    from advattacks.wrapper import Wrapper


class PGD(Attack):
    """Projected Gradient Descent attack with round-robin model loading.

    Implements the iterative attack:
        1.g_t = ∇_x L(x_t)
        2.x̃_{t+1} = x_t - α * sign(g_t)
        3.x_{t+1} = clip(x̃_{t+1}, x_0 - ε, x_0 + ε) ∩ [0, 1]

    Uses teacher-forcing loss to optimize for target prefix generation across
    multiple vision-language models with round-robin loading strategy.

    Args:
        wrappers: Sequence of model wrappers to attack.
        epsilon: Maximum L-infinity perturbation (default: 32/255).
        alpha: Step size for each PGD iteration (default: 2/255).
        num_rounds: Number of complete rounds through all models.
        steps_per_model: Number of PGD steps per model per round.
        lambda_inf: Weight for L-infinity regularization (default: 0.01).
        prefixes: Sequence of target prefix strings for jailbreak.
    """

    def __init__(
        self,
        wrappers: t.Sequence[Wrapper],
        epsilon: float = 32 / 255,
        alpha: float = 2 / 255,
        num_rounds: int = 4,
        steps_per_model: int = 5,
        lambda_inf: float = 0.01,
        prefixes: t.Sequence[str] | None = None,
    ):
        super().__init__(wrappers, epsilon, prefixes)
        self.alpha = alpha
        self.num_rounds = num_rounds
        self.steps_per_model = steps_per_model
        self.lambda_inf = lambda_inf

        if self.prefixes is None:
            self.prefixes = DEFAULT_PREFIXES

    def _compute_composite_loss(
        self,
        image: torch.Tensor,
        original_image: torch.Tensor,
        question: str,
        wrapper: Wrapper,
    ) -> torch.Tensor:
        """Compute composite loss combining teacher-forcing and
        L-infinity penalty.

        The composite loss encourages target prefix generation while penalizing
        large perturbations:
            L = (1/K) * Σ L_tf(prefix_k) + λ_inf * ||x - x_0||_inf

        Uses proper tensor arithmetic to maintain gradient flow throughout
        the computation. Each prefix contributes to the total loss via
        differentiable tensor operations.

        Args:
            image: Current perturbed image.
            original_image: Original clean image.
            question: Input question/prompt.
            wrapper: Model wrapper to compute loss with.

        Returns:
            Scalar composite loss tensor with maintained gradient flow.
        """
        # Initialize total loss as tensor (not scalar) to maintain gradients
        total_tfloss = torch.tensor(0.0, device=image.device)
        valid_prefixes = 0

        # Compute teacher-forcing loss for each target prefix
        for prefix in self.prefixes:
            try:
                tfloss = wrapper.compute_tfloss(image, question, prefix)
                # Keep tensor arithmetic - do NOT use .item()!
                total_tfloss = total_tfloss + tfloss
                valid_prefixes += 1

            except torch.cuda.OutOfMemoryError:
                print(f"Warning: OOM for prefix '{prefix}', skipping...")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"Warning: Failed to compute loss for prefix '{prefix}': {e}")
                continue

        # Average teacher-forcing loss (maintain tensor operations)
        if valid_prefixes > 0:
            avg_tfloss = total_tfloss / valid_prefixes
        else:
            avg_tfloss = torch.tensor(0.0, device=image.device)

        # L-infinity regularization penalty
        linf_penalty = torch.norm(image - original_image, p=float("inf"))

        # Composite loss with proper weighting
        return avg_tfloss + self.lambda_inf * linf_penalty

    def _pgd_step(
        self,
        image: torch.Tensor,
        original_image: torch.Tensor,
        question: str,
        wrapper: Wrapper,
    ) -> torch.Tensor:
        """Perform a single PGD update step with proper gradient
        computation.

        Computes gradient of composite loss w.r.t.image and applies PGD update
        with projection to epsilon-ball and valid pixel range. Includes gradient
        debugging information to monitor optimization progress.

        Args:
            image: Current perturbed image.
            original_image: Original clean image.
            question: Input question/prompt.
            wrapper: Model wrapper to compute gradient against.

        Returns:
            Updated perturbed image after PGD step.
        """
        # Enable gradient computation for image
        image_adv = image.clone().detach().requires_grad_(True)

        try:
            # Compute composite loss with maintained gradient flow
            loss = self._compute_composite_loss(image_adv, original_image, question, wrapper)

            # Debug: Check loss properties
            # print(f"Loss value: {loss.item():.6f}, requires_grad: {loss.requires_grad}")

            # Compute gradient w.r.t. adversarial image
            grad = torch.autograd.grad(loss, image_adv, create_graph=False)[0]

            # Debug: Check gradient properties
            grad_norm = torch.norm(grad).item()
            # print(f"Gradient norm: {grad_norm:.6f}")
            # if grad_norm < 1e-8:
            #     print("WARNING: Gradient is near zero - no effective update!")

            # PGD update: x - alpha * sign(∇L)
            normalized_grad = utils.normalize_gradient(grad)
            image_updated = image_adv - self.alpha * normalized_grad

            # Project back to epsilon-ball and valid pixel range [0, 1]
            image_projected = utils.project_linf(
                image_updated,
                original_image,
                self.epsilon,
            )

            result = image_projected.detach()

        finally:
            # Clean up intermediate variables
            if "loss" in locals():
                del loss
            if "grad" in locals():
                del grad
            if "image_adv" in locals():
                del image_adv
            if "normalized_grad" in locals():
                del normalized_grad
            if "image_updated" in locals():
                del image_updated
            if "image_projected" in locals():
                del image_projected

        return result

    def attack(self, image: torch.Tensor, text: str, verbose: bool = True) -> torch.Tensor:
        """Run PGD attack with round-robin model loading.

        Performs multi-round optimization across all models using sequential
        loading to minimize memory usage.Each model is loaded, optimized for
        several steps, then unloaded before moving to the next.

        The attack alternates between models to encourage cross-model transferable
        adversarial perturbations while maintaining memory efficiency through
        the round-robin loading strategy.

        Args:
            image: Original clean image tensor of shape (C, H, W) in [0, 1] range.
            text: Text prompt/question.
            verbose: Whether to print progress information.

        Returns:
            Adversarial image tensor with same shape as input, satisfying
            L-infinity constraint and pixel value bounds.
        """
        original_image = image.clone().detach()
        x = original_image.clone()

        total_steps = self.num_rounds * len(self.wrappers) * self.steps_per_model
        current_step = 0

        for round_idx in range(self.num_rounds):
            if verbose:
                print(f"\n=== Round {round_idx + 1}/{self.num_rounds} ===")

            for wrapper_idx, wrapper in enumerate(self.wrappers):
                # Load model into memory
                if verbose:
                    print(f"  Loading model {wrapper_idx + 1}/{len(self.wrappers)}...")
                wrapper.load()

                # Perform PGD steps for this model
                for _ in range(self.steps_per_model):
                    x_prev = x.clone()
                    x = self._pgd_step(x, original_image, text, wrapper)
                    current_step += 1

                    if verbose:
                        linf_norm = torch.norm(x - original_image, p=float("inf")).item()
                        step_change = torch.norm(x - x_prev, p=float("inf")).item()
                        print(
                            f"    Step {current_step}/{total_steps} | "
                            f"L∞: {linf_norm:.6f} | "
                            f"Δ: {step_change:.6f}"
                        )

                # Unload model and free memory
                if verbose:
                    print(f"  Unloading model {wrapper_idx + 1}...")
                wrapper.unload()

        # Final statistics
        if verbose:
            final_linf = torch.norm(x - original_image, p=float("inf")).item()
            print("\n=== Attack Complete ===")
            print(f"Final L∞ norm: {final_linf:.6f}")
            print(f"Epsilon constraint: {self.epsilon:.6f}")
            print(f"Constraint satisfied: {final_linf <= self.epsilon}")

        return x
