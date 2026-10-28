from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RunMetrics:
    coverage_pct: float
    bad2px_pct: float
    median_abs_depth_error_cm: float
    median_abs_depth_error_common_cm: float


def coverage_pct(est_valid: np.ndarray, gt_valid: np.ndarray) -> float:
    return 100.0 * np.count_nonzero(est_valid & gt_valid) / np.count_nonzero(gt_valid)


def bad_pixel_pct(
    est_disparity: np.ndarray,
    gt_disparity: np.ndarray,
    est_valid: np.ndarray,
    gt_valid: np.ndarray,
    threshold_px: float = 2.0,
) -> float:
    difference = np.abs(est_disparity - gt_disparity)
    wrong = (~est_valid) | (difference > threshold_px)
    return 100.0 * np.count_nonzero(wrong & gt_valid) / np.count_nonzero(gt_valid)


def median_abs_depth_error_cm(
    est_depth_cm: np.ndarray, gt_depth_cm: np.ndarray, mask: np.ndarray
) -> float:
    return float(np.median(np.abs(est_depth_cm[mask] - gt_depth_cm[mask])))


def evaluate_run(
    est_disparity: np.ndarray,
    est_depth_cm: np.ndarray,
    gt_disparity: np.ndarray,
    gt_depth_cm: np.ndarray,
    est_valid: np.ndarray,
    gt_valid: np.ndarray,
    baseline_valid: np.ndarray,
) -> RunMetrics:
    eval_mask = est_valid & gt_valid
    common_mask = eval_mask & baseline_valid
    return RunMetrics(
        coverage_pct=coverage_pct(est_valid, gt_valid),
        bad2px_pct=bad_pixel_pct(est_disparity, gt_disparity, est_valid, gt_valid),
        median_abs_depth_error_cm=median_abs_depth_error_cm(est_depth_cm, gt_depth_cm, eval_mask),
        median_abs_depth_error_common_cm=median_abs_depth_error_cm(
            est_depth_cm, gt_depth_cm, common_mask
        ),
    )
