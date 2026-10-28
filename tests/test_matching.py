import cv2
import numpy as np

from neural_stereo.config import SgbmSettings
from neural_stereo.matching import build_matcher, compute_disparity, rounded_num_disparities


def test_rounded_num_disparities() -> None:
    assert rounded_num_disparities(70) == 80
    assert rounded_num_disparities(64) == 64
    assert rounded_num_disparities(1) == 16


def test_recovers_known_shift(rng: np.random.Generator) -> None:
    true_disparity = 24
    texture = rng.integers(0, 255, size=(160, 240), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (0, 0), 1.5)
    left = np.dstack([texture] * 3)
    right = np.dstack([np.roll(texture, -true_disparity, axis=1)] * 3)
    matcher = build_matcher(48, SgbmSettings())
    disparity = compute_disparity(matcher, left, right)
    central = disparity[40:120, 60:180]
    estimated = float(np.nanmedian(central))
    assert abs(estimated - true_disparity) <= 1.0


def test_invalid_pixels_are_nan(rng: np.random.Generator) -> None:
    texture = rng.integers(0, 255, size=(96, 128), dtype=np.uint8)
    left = np.dstack([texture] * 3)
    right = np.dstack([np.roll(texture, -20, axis=1)] * 3)
    matcher = build_matcher(32, SgbmSettings())
    disparity = compute_disparity(matcher, left, right)
    assert np.isnan(disparity[:, :4]).all()
