from __future__ import annotations

import unittest.mock as mock

import pytest
import torch
import torch.nn as nn

from advattacks.wrapper import Processor
from advattacks.wrapper import Tokenizer
from advattacks.wrapper import Wrapper


class MockProcessor(Processor):
    def __init__(self):
        self.tokenizer = mock.MagicMock()
        self.tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    def __call__(self, images, text, **kwargs):
        # Preserve gradient tracking for images
        if images is not None:
            pixel_values = images.unsqueeze(0) if images.ndim == 3 else images
        else:
            pixel_values = torch.randn(1, 3, 224, 224)

        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "pixel_values": pixel_values,
        }

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return "This is a mocked response."


class MockTokenizer(Tokenizer):
    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return "This is a mocked response."


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Add a simple learnable parameter to create computation graph
        self.linear = nn.Linear(10, 10)
        self.image_proj = nn.Conv2d(3, 16, kernel_size=3, padding=1)

    def forward(self, pixel_values=None, _input_ids=None, labels=None, **_kwargs):
        """Mock forward pass with actual computation."""
        batch_size = 1
        seq_len = 10
        vocab_size = 32000

        # Create logits with gradient tracking
        # Use a simple computation that depends on pixel_values if provided
        if pixel_values is not None and pixel_values.requires_grad:
            # Process image to create dependency
            img_features = self.image_proj(
                pixel_values.unsqueeze(0) if pixel_values.ndim == 3 else pixel_values
            )
            img_scalar = img_features.mean()  # Reduce to scalar
            logits = (
                torch.randn(batch_size, seq_len, vocab_size, requires_grad=True) * 0.1
                + img_scalar * 0.01
            )
        else:
            logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)

        # Compute loss if labels provided
        if labels is not None:
            # Simple mock loss that depends on logits
            # Create a proper cross-entropy style loss
            # Flatten logits and create dummy targets based on labels length
            num_tokens = labels.shape[0] if labels.ndim == 1 else labels.shape[1]

            # Use actual computation that depends on both logits and image
            loss = logits[:, :num_tokens, :].mean() + 2.5  # Base loss of 2.5 + variation
        else:
            loss = None

        # Create a simple namespace to mimic model outputs
        class Outputs:
            pass

        outputs = Outputs()
        outputs.loss = loss
        outputs.logits = logits

        return outputs

    def generate(self, **_kwargs):
        """Mock generation."""
        return torch.tensor([[1, 2, 3, 4, 5]])


class MockWrapper(Wrapper):
    def __init__(self, model_path: str = "mock/model", device: torch.device | None = None):
        super().__init__(model_path, device)
        self._load_count = 0
        self._unload_count = 0

    def load(self) -> None:
        if self._is_loaded:
            return
        self.model = MockModel()
        self.processor = MockProcessor()
        self.tokenizer = MockTokenizer()
        self._is_loaded = True
        self._load_count += 1

    def unload(self) -> None:
        if not self._is_loaded:
            return
        self.model = None
        self.processor = None
        self._is_loaded = False
        self._unload_count += 1

    def prepare_inputs(self, image: torch.Tensor, text: str) -> dict[str, torch.Tensor]:
        if not self._is_loaded:
            raise RuntimeError("Model not loaded.  Call load() first.")
        return self.processor(images=image, text=text)

    def forward(
        self,
        image: torch.Tensor,
        text: str,
        target_ids: t.Optional[torch.Tensor] = None,  # noqa
    ) -> t.Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:  # noqa
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.prepare_inputs(image, text)
        if target_ids is not None:
            inputs["labels"] = target_ids

        outputs = self.model(**inputs)

        if target_ids is not None:
            return outputs.loss, outputs.logits
        return outputs.logits

    def compute_loss(
        self,
        image: torch.Tensor,
        text: str,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        loss, _ = self.forward(image, text, target_ids)
        return loss

    def compute_gradient(
        self,
        image: torch.Tensor,
        text: str,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        image_adv = image.clone().detach().requires_grad_(True)
        loss = self.compute_loss(image_adv, text, target_ids)
        return torch.autograd.grad(loss, image_adv)[0]

    def generate(
        self,
        image: torch.Tensor,
        text: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        if not self._is_loaded:
            raise RuntimeError("Model not loaded.Call load() first.")
        inputs = self.prepare_inputs(image, text)
        generate_ids = self.model.generate(**inputs)
        return self.processor.decode(generate_ids[0], skip_special_tokens=True)


@pytest.fixture
def mock_wrapper():
    """Fixture providing a mock wrapper."""
    return MockWrapper()


@pytest.fixture
def mock_wrappers():
    """Fixture providing multiple mock wrappers."""
    return [
        MockWrapper("mock/model1"),
        MockWrapper("mock/model2"),
        MockWrapper("mock/model3"),
    ]


@pytest.fixture
def sample_image():
    """Fixture providing a sample image tensor."""
    return torch.rand(3, 224, 224)  # (C, H, W) in [0, 1]


@pytest.fixture
def original_image():
    """Fixture providing an original image for perturbation tests."""
    return torch.rand(3, 224, 224)


@pytest.fixture
def sample_text():
    """Fixture providing sample text."""
    return "Can you help me with something?"


@pytest.fixture
def sample_target_ids():
    """Fixture providing sample target token IDs."""
    return torch.tensor([1, 2, 3, 4, 5])
