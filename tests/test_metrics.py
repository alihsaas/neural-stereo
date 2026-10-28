import numpy as np

from neural_stereo.calib import Calibration
from neural_stereo.depth import disparity_to_depth_cm, valid_mask
from neural_stereo.metrics import bad_pixel_pct, coverage_pct, evaluate_run

CALIB = Calibration(
    focal_length_px=1000.0,
    cx0=370.0,
    cy0=250.0,
    doffs=0.0,
    baseline_mm=100.0,
    width=740,
    height=500,
    ndisp=64,
)


def test_coverage_counts_only_gt_valid_pixels() -> None:
    gt_valid = np.array([[True, True], [True, False]])
    est_valid = np.array([[True, False], [True, True]])
    assert coverage_pct(est_valid, gt_valid) == 100.0 * 2 / 3


def test_bad_pixel_counts_invalid_estimates_as_bad() -> None:
    gt_disparity = np.full((2, 2), 10.0, dtype=np.float32)
    est_disparity = np.array([[10.0, 14.0], [np.nan, 11.0]], dtype=np.float32)
    rate = bad_pixel_pct(
        est_disparity,
        gt_disparity,
        valid_mask(est_disparity),
        valid_mask(gt_disparity),
    )
    assert rate == 100.0 * 2 / 4


def test_common_mask_excludes_pixels_dropped_by_baseline() -> None:
    gt_disparity = np.full((2, 2), 10.0, dtype=np.float32)
    est_disparity = np.array([[12.0, 20.0], [np.nan, 11.0]], dtype=np.float32)
    gt_depth = disparity_to_depth_cm(gt_disparity, CALIB)
    est_depth = disparity_to_depth_cm(est_disparity, CALIB)
    est_valid = valid_mask(est_disparity)
    gt_valid = valid_mask(gt_disparity)
    baseline_valid = np.array([[True, False], [True, True]])
    metrics = evaluate_run(
        est_disparity, est_depth, gt_disparity, gt_depth, est_valid, gt_valid, baseline_valid
    )
    errors = np.abs(est_depth - gt_depth)
    plain_expected = float(np.median([errors[0, 0], errors[0, 1], errors[1, 1]]))
    common_expected = float(np.median([errors[0, 0], errors[1, 1]]))
    assert metrics.median_abs_depth_error_cm == plain_expected
    assert metrics.median_abs_depth_error_common_cm == common_expected
    assert metrics.median_abs_depth_error_common_cm < metrics.median_abs_depth_error_cm
