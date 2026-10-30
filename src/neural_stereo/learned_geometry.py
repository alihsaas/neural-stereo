import numpy as np


def pixel_grid(width: int, height: int) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1)


def correspondences_to_disparity(
    grid: np.ndarray,
    predicted: np.ndarray,
    match_valid: np.ndarray,
    width: int,
    height: int,
    max_disparity: float,
    max_row_deviation_px: float,
) -> tuple[np.ndarray, float]:
    disparity_values = grid[:, 0] - predicted[:, 0]
    row_deviation = np.abs(predicted[:, 1] - grid[:, 1])
    valid = (
        match_valid
        & (row_deviation <= max_row_deviation_px)
        & (disparity_values > 0.0)
        & (disparity_values < max_disparity)
    )
    disparity = np.full(height * width, np.nan, dtype=np.float32)
    disparity[valid] = disparity_values[valid]
    if np.any(match_valid):
        mean_row_deviation = float(np.mean(row_deviation[match_valid]))
    else:
        mean_row_deviation = float("nan")
    return disparity.reshape(height, width), mean_row_deviation
