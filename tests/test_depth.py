import numpy as np

from neural_stereo.calib import Calibration
from neural_stereo.depth import disparity_to_depth_cm, valid_mask


def make_calib(doffs: float) -> Calibration:
    return Calibration(
        focal_length_px=1000.0,
        cx0=370.0,
        cy0=250.0,
        doffs=doffs,
        baseline_mm=100.0,
        width=740,
        height=500,
        ndisp=64,
    )


def test_depth_formula_includes_doffs() -> None:
    disparity = np.array([[90.0]], dtype=np.float32)
    with_offset = disparity_to_depth_cm(disparity, make_calib(10.0))
    without_offset = disparity_to_depth_cm(disparity, make_calib(0.0))
    np.testing.assert_allclose(with_offset, [[100.0]])
    np.testing.assert_allclose(without_offset, [[100000.0 / 90.0 / 10.0]])


def test_invalid_disparity_gives_nan_depth() -> None:
    disparity = np.array([[np.nan, 50.0]], dtype=np.float32)
    depth = disparity_to_depth_cm(disparity, make_calib(10.0))
    assert np.isnan(depth[0, 0])
    assert np.isfinite(depth[0, 1])


def test_valid_mask() -> None:
    disparity = np.array([[np.nan, 0.0, -1.0, 12.5]], dtype=np.float32)
    np.testing.assert_array_equal(valid_mask(disparity), [[False, False, False, True]])
