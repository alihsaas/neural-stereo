import numpy as np

from .calib import Calibration


def disparity_to_depth_cm(disparity: np.ndarray, calib: Calibration) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        depth_mm = calib.baseline_mm * calib.focal_length_px / (disparity + calib.doffs)
    return depth_mm / 10.0


def valid_mask(disparity: np.ndarray) -> np.ndarray:
    return np.isfinite(disparity) & (disparity > 0.0)
