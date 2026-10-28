import cv2
import numpy as np

from .calib import Calibration
from .config import SgbmSettings


def rounded_num_disparities(ndisp: int) -> int:
    return ((ndisp + 15) // 16) * 16


def build_matcher(num_disparities: int, settings: SgbmSettings) -> cv2.StereoSGBM:
    return cv2.StereoSGBM.create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=settings.block_size,
        P1=settings.p1,
        P2=settings.p2,
        disp12MaxDiff=settings.disp12_max_diff,
        uniquenessRatio=settings.uniqueness_ratio,
        speckleWindowSize=settings.speckle_window_size,
        speckleRange=settings.speckle_range,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def sgbm_disparity_factory(calib: Calibration, settings: SgbmSettings):
    matcher = build_matcher(rounded_num_disparities(calib.ndisp), settings)

    def estimate(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        return compute_disparity(matcher, left_bgr, right_bgr)

    return estimate


def compute_disparity(
    matcher: cv2.StereoSGBM, left_bgr: np.ndarray, right_bgr: np.ndarray
) -> np.ndarray:
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    raw = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disparity = raw.copy()
    disparity[raw <= 0.0] = np.nan
    return disparity
