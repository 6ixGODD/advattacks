from __future__ import annotations

import abc
import os
import pathlib
import types
import typing as t

import torch
import torch.nn as nn


class Processor(t.Protocol):
    """Protocol for model processors."""

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> dict[str, torch.Tensor]: ...

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str: ...


class Tokenizer(t.Protocol):
    """Protocol for model tokenizers."""

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> dict[str, torch.Tensor]: ...

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str: ...


class Wrapper(abc.ABC):
    """Abstract base class for vision-language model wrappers.

    Manages model lifecycle, provides unified interface for forward pass,
    gradient computation, and memory management.

    Attributes:
        model: The underlying vision-language model.
        processor: The model's processor for input preparation and decoding.
        tokenizer: The model's tokenizer for text processing.

    Args:
        model_path: Path to the pretrained model.
        device: Device to load model on.  Defaults to CUDA if available.
    """

    model: nn.Module | None
    processor: Processor | None
    tokenizer: Tokenizer | None
    _is_loaded: bool

    def __init__(self, model_path: str | os.PathLike[str], device: torch.device | None = None):
        self.model_path = pathlib.Path(model_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._is_loaded = False

    @abc.abstractmethod
    def load(self) -> None:
        """Load model and processor into memory."""

    def unload(self) -> None:
        """Unload model and free GPU memory."""
        if not self._is_loaded:
            return

        del self.model
        del self.processor
        del self.tokenizer
        self.model = None
        self.processor = None
        self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._is_loaded = False

    @abc.abstractmethod
    def prepare_inputs(self, image: torch.Tensor, text: str) -> dict[str, torch.Tensor]:
        """Prepare inputs for the model.

        Args:
            image: Image tensor of shape (C, H, W) in [0, 1] range.
            text: Text prompt.

        Returns:
            Dictionary of model inputs.
        """

    @abc.abstractmethod
    def prepare_tf_inputs(
        self,
        image: torch.Tensor,
        question: str,
        target_prefix: str,
    ) -> dict[str, torch.Tensor]:
        """Prepare inputs for teacher-forcing loss computation.

        Args:
            image: Image tensor of shape (C, H, W) in [0, 1] range.
            question: Question/prompt text.
            target_prefix: Target prefix string (e.g., "Sure, here is how to").

        Returns:
            Dictionary of model inputs.
        """

    @abc.abstractmethod
    def forward(
        self,
        image: torch.Tensor,
        text: str,
        target_ids: t.Optional[torch.Tensor] = None,  # noqa
    ) -> t.Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:  # noqa
        """Forward pass through the model.

        Args:
            image: Image tensor of shape (C, H, W) in [0, 1] range.
            text: Text prompt.
            target_ids: Target token IDs for loss computation.  If None, returns logits only.

        Returns:
            If target_ids is None: logits tensor.
            If target_ids is provided: tuple of (loss, logits).
        """

    @abc.abstractmethod
    def compute_tfloss(
        self,
        image: torch.Tensor,
        question: str,
        target_prefix: str,
    ) -> torch.Tensor:
        """Compute teacher-forcing loss for given inputs and target
        prefix.

        Args:
            image: Image tensor of shape (C, H, W) in [0, 1] range.
            question: Question/prompt text.
            target_prefix: Target prefix string (e.g., "Sure, here is how to").

        Returns:
            Scalar teacher-forcing loss tensor.
        """

    def compute_loss(
        self,
        image: torch.Tensor,
        text: str,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for given inputs and targets.

        Args:
            image: Image tensor of shape (C, H, W) in [0, 1] range.
            text: Text prompt.
            target_ids: Target token IDs of shape (seq_len,).

        Returns:
            Scalar loss tensor.
        """
        loss, _ = self.forward(image, text, target_ids)
        return loss

    def compute_gradient(
        self,
        image: torch.Tensor,
        text: str,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient of loss w.r.t. image.

        Args:
            image: Image tensor of shape (C, H, W) in [0, 1] range.
            text: Text prompt.
            target_ids: Target token IDs.

        Returns:
            Gradient tensor of same shape as image.
        """
        image_adv = image.clone().detach().requires_grad_(True)
        loss = self.compute_loss(image_adv, text, target_ids)
        return torch.autograd.grad(loss, image_adv)[0]

    def compute_tf_gradient(
        self,
        image: torch.Tensor,
        question: str,
        target_prefix: str,
    ) -> torch.Tensor:
        """Compute gradient of teacher-forcing loss w.r.t. image.

        Args:
            image: Image tensor of shape (C, H, W) in [0, 1] range.
            question: Question/prompt text.
            target_prefix: Target prefix string.

        Returns:
            Gradient tensor of same shape as image.
        """
        image_adv = image.clone().detach().requires_grad_(True)
        loss = self.compute_tfloss(image_adv, question, target_prefix)
        return torch.autograd.grad(loss, image_adv)[0]

    def generate(
        self,
        image: torch.Tensor,
        text: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        """Generate text response given image and text prompt.

        Args:
            image: Image tensor of shape (C, H, W) in [0, 1] range.
            text: Text prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text string.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.prepare_inputs(image, text)

        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
            )

        return self.processor.decode(generate_ids[0], skip_special_tokens=True)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def __enter__(self) -> t.Self:
        self.load()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.unload()
