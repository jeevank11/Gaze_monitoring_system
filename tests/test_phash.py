"""Tests for perceptual hashing (screen content only, never faces)."""

from __future__ import annotations

import numpy as np

from gaze_analytics.content.phash import compute_phash, hamming


def _solid(color: tuple[int, int, int], size: tuple[int, int] = (128, 128)) -> np.ndarray:
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def test_phash_is_a_64bit_int() -> None:
    img = _solid((10, 20, 30))
    ph = compute_phash(img)
    assert isinstance(ph, int)
    assert 0 <= ph < (1 << 64)


def test_identical_images_hash_identically() -> None:
    img = np.random.default_rng(42).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    assert compute_phash(img) == compute_phash(img.copy())


def test_hamming_distance_small_for_similar_images() -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    b = a.copy()
    # Small perturbation — a few pixels flipped.
    b[0:5, 0:5] = 255
    dist = hamming(compute_phash(a), compute_phash(b))
    assert dist <= 12, f"expected small hamming distance, got {dist}"


def test_hamming_distance_large_for_different_images() -> None:
    # A high-frequency checkerboard vs. pure noise — DCT coefficients diverge
    # dramatically, forcing a large Hamming distance.
    n = 256
    board = np.indices((n, n)).sum(axis=0) % 2
    checker = (board * 255).astype(np.uint8)
    checker_rgb = np.stack([checker, checker, checker], axis=-1)

    rng = np.random.default_rng(123)
    noise = rng.integers(0, 256, (n, n, 3), dtype=np.uint8)

    dist = hamming(compute_phash(checker_rgb), compute_phash(noise))
    assert dist >= 12, f"expected large hamming distance, got {dist}"


def test_hamming_symmetry_and_zero() -> None:
    assert hamming(0, 0) == 0
    assert hamming(0xDEADBEEF, 0x00000000) == hamming(0x00000000, 0xDEADBEEF)
