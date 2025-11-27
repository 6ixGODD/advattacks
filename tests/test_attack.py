"""Tests for attack algorithms."""

from __future__ import annotations

import torch

from advattacks.attack.pgd import PGD
from tests.conftest import MockWrapper


class TestPGD:
    """Test PGD attack."""

    def test_pgd_initialization(self):
        """Test PGD is properly initialized."""
        wrappers = [MockWrapper("model1"), MockWrapper("model2")]
        attack = PGD(
            wrappers=wrappers,
            epsilon=32 / 255,
            alpha=2 / 255,
            num_rounds=2,
            steps_per_model=3,
        )

        assert attack.epsilon == 32 / 255
        assert attack.alpha == 2 / 255
        assert attack.num_rounds == 2
        assert attack.steps_per_model == 3
        assert len(attack.wrappers) == 2

    def test_pgd_attack_returns_perturbed_image(self, sample_image, sample_text):
        """Test PGD returns perturbed image."""
        wrappers = [MockWrapper("model1")]
        attack = PGD(
            wrappers=wrappers,
            epsilon=32 / 255,
            alpha=2 / 255,
            num_rounds=1,
            steps_per_model=2,
        )

        adv_image = attack(sample_image, sample_text, verbose=False)

        assert isinstance(adv_image, torch.Tensor)
        assert adv_image.shape == sample_image.shape

    def test_pgd_attack_satisfies_epsilon_constraint(self, sample_image, sample_text):
        """Test PGD satisfies epsilon constraint."""
        epsilon = 32 / 255
        wrappers = [MockWrapper("model1")]
        attack = PGD(
            wrappers=wrappers,
            epsilon=epsilon,
            alpha=2 / 255,
            num_rounds=2,
            steps_per_model=3,
        )

        adv_image = attack(sample_image, sample_text, verbose=False)

        linf_norm = torch.norm(adv_image - sample_image, p=float("inf"))
        assert linf_norm <= epsilon + 1e-6

    def test_pgd_attack_produces_perturbation(self, sample_image, sample_text):
        """Test PGD produces non-zero perturbation."""
        wrappers = [MockWrapper("model1")]
        attack = PGD(
            wrappers=wrappers,
            epsilon=32 / 255,
            alpha=2 / 255,
            num_rounds=3,  # Increase rounds to ensure perturbation
            steps_per_model=5,  # Increase steps
        )

        adv_image = attack(sample_image, sample_text, verbose=False)

        # Should have some perturbation (use a tolerance for numerical stability)
        linf_diff = torch.norm(adv_image - sample_image, p=float("inf"))
        assert linf_diff > 1e-6, f"Expected perturbation but got L∞ difference of {linf_diff}"

    def test_pgd_attack_valid_pixel_range(self, sample_image, sample_text):
        """Test PGD keeps pixels in [0, 1] range."""
        wrappers = [MockWrapper("model1")]
        attack = PGD(
            wrappers=wrappers,
            epsilon=32 / 255,
            alpha=2 / 255,
            num_rounds=2,
            steps_per_model=3,
        )

        adv_image = attack(sample_image, sample_text, verbose=False)

        assert torch.all(adv_image >= 0.0)
        assert torch.all(adv_image <= 1.0)

    def test_pgd_attack_multiple_rounds(self, sample_image, sample_text):
        """Test PGD runs multiple rounds correctly."""
        wrappers = [MockWrapper("model1"), MockWrapper("model2")]
        attack = PGD(
            wrappers=wrappers,
            epsilon=32 / 255,
            alpha=2 / 255,
            num_rounds=3,
            steps_per_model=2,
        )

        adv_image = attack(sample_image, sample_text, verbose=False)

        assert isinstance(adv_image, torch.Tensor)
        # Should have run 3 rounds * 2 models * 2 steps = 12 total steps

    def test_pgd_tokenized_cache(self, sample_image, sample_text):
        """Test PGD caches tokenized prefixes."""
        wrappers = [MockWrapper("model1")]
        attack = PGD(wrappers=wrappers, num_rounds=2, steps_per_model=2)

        # Cache should be empty initially
        assert len(attack._tokenized_cache) == 0

        attack(sample_image, sample_text, verbose=False)

        # Cache should be populated after attack
        assert len(attack._tokenized_cache) > 0

    def test_pgd_unloads_models(self, sample_image, sample_text):
        """Test PGD unloads models after each round."""
        wrappers = [MockWrapper("model1"), MockWrapper("model2")]
        attack = PGD(wrappers=wrappers, num_rounds=1, steps_per_model=2)

        attack(sample_image, sample_text, verbose=False)

        # All wrappers should be unloaded after attack
        for wrapper in wrappers:
            assert not wrapper.is_loaded
