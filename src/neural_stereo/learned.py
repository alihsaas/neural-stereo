import cv2
import numpy as np
import torch
from flexglue import FlexGlue

from .calib import Calibration
from .learned_geometry import correspondences_to_disparity, pixel_grid
from .matching import rounded_num_disparities

MAX_ROW_DEVIATION_PX = 2.0


def bgr_to_tensor(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor[None].to(device)


class FlexGlueDisparityEstimator:
    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        self.model = FlexGlue(refine=True).eval().to(self.device)
        self.last_diagnostics: dict[str, float] = {}

    def estimate(
        self, left_bgr: np.ndarray, right_bgr: np.ndarray, max_disparity: float
    ) -> np.ndarray:
        height, width = left_bgr.shape[:2]
        grid = pixel_grid(width, height)
        inputs0 = {
            "image": bgr_to_tensor(left_bgr, self.device),
            "keypoints": torch.from_numpy(grid)[None].to(self.device),
        }
        inputs1 = {"image": bgr_to_tensor(right_bgr, self.device)}
        with torch.inference_mode():
            prediction = self.model.match(inputs0, inputs1)
        predicted = prediction["p2d0_i1"][0].float().cpu().numpy()
        match_valid = prediction["valid0"][0].cpu().numpy().astype(bool)
        disparity, mean_row_deviation = correspondences_to_disparity(
            grid, predicted, match_valid, width, height, max_disparity, MAX_ROW_DEVIATION_PX
        )
        self.last_diagnostics = {
            "covisibility_score": float(prediction["covisibility_score"][0]),
            "mean_abs_row_deviation_px": mean_row_deviation,
        }
        return disparity


def flexglue_disparity_factory(estimator: FlexGlueDisparityEstimator, calib: Calibration):
    max_disparity = float(rounded_num_disparities(calib.ndisp))

    def estimate(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        return estimator.estimate(left_bgr, right_bgr, max_disparity)

    return estimate
