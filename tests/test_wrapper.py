from __future__ import annotations

import pytest
import torch


class TestWrapper:
    def test_wrapper_load(self, mock_wrapper):
        """Test wrapper loads model correctly."""
        mock_wrapper.load()
        assert mock_wrapper.is_loaded
        assert mock_wrapper.model is not None
        assert mock_wrapper.processor is not None

    def test_wrapper_load_idempotent(self, mock_wrapper):
        """Test multiple load calls are idempotent."""
        mock_wrapper.load()
        model_ref = mock_wrapper.model

        mock_wrapper.load()
        assert mock_wrapper.model is model_ref

    def test_wrapper_unload(self, mock_wrapper):
        """Test wrapper unloads model correctly."""
        mock_wrapper.load()
        mock_wrapper.unload()

        assert not mock_wrapper.is_loaded
        assert mock_wrapper.model is None
        assert mock_wrapper.processor is None

    def test_wrapper_unload_idempotent(self, mock_wrapper):
        """Test multiple unload calls are idempotent."""
        mock_wrapper.load()
        mock_wrapper.unload()
        mock_wrapper.unload()  # Should not raise

        assert not mock_wrapper.is_loaded

    def test_wrapper_context_manager(self, mock_wrapper):
        """Test wrapper works as context manager."""
        assert not mock_wrapper.is_loaded

        with mock_wrapper as wrapper:
            assert wrapper.is_loaded
            assert wrapper.model is not None

        assert not mock_wrapper.is_loaded

    def test_prepare_inputs_requires_load(self, mock_wrapper, sample_image, sample_text):
        """Test prepare_inputs raises if model not loaded."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            mock_wrapper.prepare_inputs(sample_image, sample_text)

    def test_prepare_inputs(self, mock_wrapper, sample_image, sample_text):
        """Test prepare_inputs returns correct format."""
        mock_wrapper.load()
        inputs = mock_wrapper.prepare_inputs(sample_image, sample_text)

        assert isinstance(inputs, dict)
        assert "input_ids" in inputs
        assert isinstance(inputs["input_ids"], torch.Tensor)

    def test_forward_without_targets(self, mock_wrapper, sample_image, sample_text):
        """Test forward pass without target IDs returns logits."""
        mock_wrapper.load()
        logits = mock_wrapper.forward(sample_image, sample_text)

        assert isinstance(logits, torch.Tensor)
        assert logits.requires_grad

    def test_forward_with_targets(self, mock_wrapper, sample_image, sample_text, sample_target_ids):
        """Test forward pass with target IDs returns loss and logits."""
        mock_wrapper.load()
        loss, logits = mock_wrapper.forward(sample_image, sample_text, sample_target_ids)

        assert isinstance(loss, torch.Tensor)
        assert isinstance(logits, torch.Tensor)
        assert loss.requires_grad
        assert logits.requires_grad

    def test_compute_loss(self, mock_wrapper, sample_image, sample_text, sample_target_ids):
        """Test compute_loss returns scalar loss."""
        mock_wrapper.load()
        loss = mock_wrapper.compute_loss(sample_image, sample_text, sample_target_ids)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.requires_grad

    def test_compute_gradient(self, mock_wrapper, sample_image, sample_text, sample_target_ids):
        """Test compute_gradient returns gradient of correct shape."""
        mock_wrapper.load()
        grad = mock_wrapper.compute_gradient(sample_image, sample_text, sample_target_ids)

        assert isinstance(grad, torch.Tensor)
        assert grad.shape == sample_image.shape
        assert not grad.requires_grad

    def test_generate(self, mock_wrapper, sample_image, sample_text):
        """Test generate returns string."""
        mock_wrapper.load()
        response = mock_wrapper.generate(sample_image, sample_text)

        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_requires_load(self, mock_wrapper, sample_image, sample_text):
        """Test generate raises if model not loaded."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            mock_wrapper.generate(sample_image, sample_text)
