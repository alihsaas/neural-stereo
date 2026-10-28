from pathlib import Path

import numpy as np
import pytest

from neural_stereo.pfm import read_pfm


def write_pfm_bytes(path: Path, array: np.ndarray, little_endian: bool) -> None:
    height, width = array.shape
    scale = -1.0 if little_endian else 1.0
    dtype = "<f4" if little_endian else ">f4"
    with open(path, "wb") as stream:
        stream.write(b"Pf\n")
        stream.write(f"{width} {height}\n".encode("ascii"))
        stream.write(f"{scale}\n".encode("ascii"))
        stream.write(np.flipud(array).astype(dtype).tobytes())


def test_reads_little_endian(tmp_path: Path) -> None:
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "sample.pfm"
    write_pfm_bytes(path, array, little_endian=True)
    result = read_pfm(path)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, array)


def test_reads_big_endian(tmp_path: Path) -> None:
    array = np.linspace(0.5, 6.0, 12, dtype=np.float32).reshape(4, 3)
    path = tmp_path / "sample.pfm"
    write_pfm_bytes(path, array, little_endian=False)
    np.testing.assert_allclose(read_pfm(path), array, rtol=1e-6)


def test_preserves_infinity(tmp_path: Path) -> None:
    array = np.array([[1.0, np.inf], [2.0, 3.0]], dtype=np.float32)
    path = tmp_path / "sample.pfm"
    write_pfm_bytes(path, array, little_endian=True)
    result = read_pfm(path)
    assert np.isinf(result[0, 1])


def test_rejects_non_pfm(tmp_path: Path) -> None:
    path = tmp_path / "sample.pfm"
    path.write_bytes(b"P6\n1 1\n255\nabc")
    with pytest.raises(ValueError, match="not a PFM file"):
        read_pfm(path)
