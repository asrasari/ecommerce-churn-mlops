import numpy as np

from src.monitor import calculate_psi


def test_psi_zero_for_identical_distribution():
    rng = np.random.default_rng(0)
    base = rng.normal(size=2000)
    assert calculate_psi(base, base) < 0.01


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, size=2000)
    shifted = rng.normal(3, 1, size=2000)
    assert calculate_psi(base, shifted) > 0.2
