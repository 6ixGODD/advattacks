from __future__ import annotations

import abc
import typing as t

import torch

if t.TYPE_CHECKING:
    from advattacks.wrapper import Wrapper


class Attack(abc.ABC):
    """Abstract base class for adversarial attacks on vision-language
    models.

    Attributes:
        wrappers: Sequence of model wrappers to attack.
        epsilon: Maximum L-infinity perturbation bound.
        prefixes: Sequence of target prefix strings for jailbreak.

    Args:
        wrappers: Sequence of model wrappers to attack.
        epsilon: Maximum L-infinity perturbation (default: 32/255).
        prefixes: Sequence of target prefix strings for jailbreak.
    """

    def __init__(
        self,
        wrappers: t.Sequence[Wrapper],
        epsilon: float = 32 / 255,
        prefixes: t.Sequence[str] | None = None,
    ):
        self.wrappers = wrappers
        self.epsilon = epsilon
        self.prefixes = prefixes

    @abc.abstractmethod
    def attack(self, image: torch.Tensor, text: str, verbose: bool = True) -> torch.Tensor:
        """Run adversarial attack.

        Args:
            image: Original clean image tensor of shape (C, H, W) in [0, 1]
                range.
            text: Text prompt.
            verbose: Whether to print progress.

        Returns:
            Adversarial image tensor.
        """

    def __call__(
        self,
        image: torch.Tensor,
        text: str,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Convenience method to run attack.

        Args:
            image: Original clean image.
            text: Text prompt.
            verbose: Whether to print progress.

        Returns:
            Adversarial image.
        """
        return self.attack(image, text, verbose)
