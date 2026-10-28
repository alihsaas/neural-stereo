import numpy as np

from neural_stereo.codec import jpeg_roundtrip, png_encoded_size


def make_image(rng: np.random.Generator) -> np.ndarray:
    gradient = np.linspace(0, 255, 96, dtype=np.uint8)
    base = np.tile(gradient, (96, 1))
    noise = rng.integers(0, 40, size=(96, 96), dtype=np.uint8)
    return np.dstack([base + noise // 4, base, base[::-1]]).astype(np.uint8)


def test_lower_quality_means_fewer_bytes(rng: np.random.Generator) -> None:
    image = make_image(rng)
    _, bytes_q95 = jpeg_roundtrip(image, 95)
    _, bytes_q20 = jpeg_roundtrip(image, 20)
    assert bytes_q20 < bytes_q95


def test_roundtrip_preserves_shape_and_dtype(rng: np.random.Generator) -> None:
    image = make_image(rng)
    decoded, byte_count = jpeg_roundtrip(image, 80)
    assert decoded.shape == image.shape
    assert decoded.dtype == np.uint8
    assert byte_count > 0


def test_png_encoded_size_positive(rng: np.random.Generator) -> None:
    assert png_encoded_size(make_image(rng)) > 0
