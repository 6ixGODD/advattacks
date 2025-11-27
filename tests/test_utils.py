from __future__ import annotations

import torch

from advattacks import utils


class TestUtils:
    def test_project_linf_no_perturbation(self):
        """Test projection with no perturbation."""
        x_original = torch.rand(3, 224, 224)
        x = x_original.clone()
        epsilon = 32 / 255

        x_proj = utils.project_linf(x, x_original, epsilon)

        assert torch.allclose(x_proj, x_original)

    def test_project_linf_within_epsilon(self):
        """Test projection with perturbation within epsilon."""
        x_original = torch.rand(3, 224, 224)
        x = x_original + 0.1  # Small perturbation
        epsilon = 32 / 255

        x_proj = utils.project_linf(x, x_original, epsilon)

        linf_norm = torch.norm(x_proj - x_original, p=float("inf"))
        assert linf_norm <= epsilon + 1e-6  # Small tolerance

    def test_project_linf_exceeds_epsilon(self):
        """Test projection clips perturbation exceeding epsilon."""
        x_original = torch.rand(3, 224, 224)
        x = x_original + 0.5  # Large perturbation
        epsilon = 32 / 255

        x_proj = utils.project_linf(x, x_original, epsilon)

        linf_norm = torch.norm(x_proj - x_original, p=float("inf"))
        assert linf_norm <= epsilon + 1e-6

    def test_project_linf_clips_to_valid_range(self):
        """Test projection clips to [0, 1] range."""
        x_original = torch.rand(3, 224, 224)
        x = x_original + 2.0  # Push out of valid range
        epsilon = 32 / 255

        x_proj = utils.project_linf(x, x_original, epsilon)

        assert torch.all(x_proj >= 0.0)
        assert torch.all(x_proj <= 1.0)

    def test_normalize_gradient_sign(self):
        """Test gradient normalization returns sign."""
        grad = torch.tensor([[-0.5, 0.3], [0.0, -0.1]])

        normalized = utils.normalize_gradient(grad)

        expected = torch.tensor([[-1.0, 1.0], [0.0, -1.0]])
        assert torch.equal(normalized, expected)

    def test_normalize_gradient_preserves_shape(self):
        """Test gradient normalization preserves shape."""
        grad = torch.randn(3, 224, 224)

        normalized = utils.normalize_gradient(grad)

        assert normalized.shape == grad.shape
