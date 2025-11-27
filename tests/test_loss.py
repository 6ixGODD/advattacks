from __future__ import annotations

import torch

from advattacks.loss import CompositeLoss
from advattacks.loss import CovertLoss
from advattacks.loss import TargetLoss


class TestCovertLoss:
    def test_covert_loss_zero(self, original_image):
        """Test covert loss is zero for identical images."""
        loss_fn = CovertLoss(original_image)
        loss = loss_fn(original_image)

        assert torch.isclose(loss, torch.tensor(0.0))

    def test_covert_loss_positive(self, original_image):
        """Test covert loss is positive for perturbed images."""
        perturbed = original_image + 0.1
        loss_fn = CovertLoss(original_image)
        loss = loss_fn(perturbed)

        assert loss > 0

    def test_covert_loss_is_linf_norm(self, original_image):
        """Test covert loss equals L-infinity norm."""
        perturbed = original_image + torch.randn_like(original_image) * 0.1
        loss_fn = CovertLoss(original_image)
        loss = loss_fn(perturbed)

        expected_linf = torch.norm(perturbed - original_image, p=float("inf"))
        assert torch.isclose(loss, expected_linf)


class TestTargetLoss:
    """Test target loss (teacher-forcing CE)."""

    def test_target_loss_single_model_single_prefix(
        self, mock_wrapper, sample_image, sample_text, sample_target_ids
    ):
        """Test target loss with single model and prefix."""
        mock_wrapper.load()
        loss_fn = TargetLoss([mock_wrapper], [sample_target_ids])

        loss = loss_fn(sample_image, sample_text)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss > 0

    def test_target_loss_multiple_models(self, sample_image, sample_text, sample_target_ids):
        """Test target loss averages across models."""
        from tests.conftest import MockWrapper

        wrappers = [MockWrapper("model1"), MockWrapper("model2")]
        for wrapper in wrappers:
            wrapper.load()

        loss_fn = TargetLoss(wrappers, [sample_target_ids])
        loss = loss_fn(sample_image, sample_text)

        assert isinstance(loss, torch.Tensor)
        assert loss > 0

    def test_target_loss_multiple_prefixes(self, mock_wrapper, sample_image, sample_text):
        """Test target loss averages across prefixes."""
        mock_wrapper.load()
        prefixes = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6]),
            torch.tensor([7, 8, 9]),
        ]

        loss_fn = TargetLoss([mock_wrapper], prefixes)
        loss = loss_fn(sample_image, sample_text)

        assert isinstance(loss, torch.Tensor)
        assert loss > 0


class TestCompositeLoss:
    """Test composite loss."""

    def test_composite_loss_combination(
        self, mock_wrapper, sample_image, sample_text, sample_target_ids, original_image
    ):
        """Test composite loss combines target and covert losses."""
        mock_wrapper.load()

        target_loss = TargetLoss([mock_wrapper], [sample_target_ids])
        covert_loss = CovertLoss(original_image)
        composite = CompositeLoss(target_loss, covert_loss, lambda_target=1.0, lambda_inf=0.01)

        loss = composite(sample_image, sample_text)

        assert isinstance(loss, torch.Tensor)
        assert loss > 0

    def test_composite_loss_weights(
        self, mock_wrapper, sample_image, sample_text, sample_target_ids, original_image
    ):
        """Test composite loss respects lambda weights."""
        mock_wrapper.load()

        target_loss = TargetLoss([mock_wrapper], [sample_target_ids])
        covert_loss = CovertLoss(original_image)

        # High target weight
        composite_high = CompositeLoss(
            target_loss, covert_loss, lambda_target=10.0, lambda_inf=0.01
        )
        loss_high = composite_high(sample_image, sample_text)

        # Low target weight
        composite_low = CompositeLoss(target_loss, covert_loss, lambda_target=0.1, lambda_inf=0.01)
        loss_low = composite_low(sample_image, sample_text)

        # Different weights should give different losses
        assert not torch.isclose(loss_high, loss_low)
