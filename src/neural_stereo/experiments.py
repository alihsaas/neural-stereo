from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .calib import Calibration, load_calib
from .codec import jpeg_roundtrip
from .config import BASELINE_CONDITION, JPEG_QUALITIES, PAIRS_PER_SECOND
from .datasets import scene_dir
from .depth import disparity_to_depth_cm, valid_mask
from .metrics import evaluate_run
from .pfm import read_pfm
from .plotting import plot_scene_grid
from .reporting import StudyRow, mbps_estimate

DisparityFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
DisparityFactory = Callable[[Calibration], DisparityFn]


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read {path}")
    return image


def run_scene(
    scene: str, disparity_factory: DisparityFactory, grid_destination: Path
) -> list[StudyRow]:
    folder = scene_dir(scene)
    calib = load_calib(folder / "calib.txt")
    gt_disparity = read_pfm(folder / "disp0GT.pfm")
    gt_disparity[~np.isfinite(gt_disparity)] = np.nan
    gt_valid = valid_mask(gt_disparity)
    gt_depth = disparity_to_depth_cm(gt_disparity, calib)
    left = load_image(folder / "im0.png")
    right = load_image(folder / "im1.png")
    estimate = disparity_factory(calib)
    conditions = (
        (BASELINE_CONDITION, None),
        *((f"q{quality}", quality) for quality in JPEG_QUALITIES),
    )
    rows: list[StudyRow] = []
    panels: list[tuple[str, np.ndarray, np.ndarray]] = []
    baseline_valid = np.zeros_like(gt_valid)
    for condition, quality in conditions:
        if quality is None:
            left_decoded, right_decoded = left, right
            bytes_pair = (folder / "im0.png").stat().st_size + (folder / "im1.png").stat().st_size
        else:
            left_decoded, left_bytes = jpeg_roundtrip(left, quality)
            right_decoded, right_bytes = jpeg_roundtrip(right, quality)
            bytes_pair = left_bytes + right_bytes
        est_disparity = estimate(left_decoded, right_decoded)
        est_valid = valid_mask(est_disparity)
        est_depth = disparity_to_depth_cm(est_disparity, calib)
        if quality is None:
            baseline_valid = est_valid
        metrics = evaluate_run(
            est_disparity, est_depth, gt_disparity, gt_depth, est_valid, gt_valid, baseline_valid
        )
        rows.append(
            StudyRow(
                scene=scene,
                condition=condition,
                jpeg_quality=quality,
                bytes_pair=bytes_pair,
                mbps_estimate_10fps=mbps_estimate(bytes_pair, PAIRS_PER_SECOND),
                coverage_pct=metrics.coverage_pct,
                bad2px_pct=metrics.bad2px_pct,
                median_abs_depth_error_cm=metrics.median_abs_depth_error_cm,
                median_abs_depth_error_common_cm=metrics.median_abs_depth_error_common_cm,
            )
        )
        abs_error = np.abs(est_depth - gt_depth)
        abs_error[~(est_valid & gt_valid)] = np.nan
        panels.append((condition, np.where(est_valid, est_depth, np.nan), abs_error))
        print(
            f"  {condition}: {bytes_pair} bytes, coverage {metrics.coverage_pct:.1f}%, "
            f"bad2px {metrics.bad2px_pct:.1f}%, median err {metrics.median_abs_depth_error_cm:.2f} cm"
        )
    depth_limits = tuple(np.percentile(gt_depth[gt_valid], (2, 98)))
    baseline_error = panels[0][2]
    error_limit = float(np.nanpercentile(baseline_error, 95))
    plot_scene_grid(
        scene,
        cv2.cvtColor(left, cv2.COLOR_BGR2RGB),
        np.where(gt_valid, gt_depth, np.nan),
        panels,
        depth_limits,
        max(error_limit, 1.0),
        grid_destination,
    )
    return rows
