import numpy as np

from neural_stereo.learned_geometry import correspondences_to_disparity, pixel_grid


def test_pixel_grid_order_and_corner_convention() -> None:
    grid = pixel_grid(3, 2)
    assert grid.shape == (6, 2)
    np.testing.assert_array_equal(grid[0], [0.5, 0.5])
    np.testing.assert_array_equal(grid[1], [1.5, 0.5])
    np.testing.assert_array_equal(grid[3], [0.5, 1.5])


def test_known_shift_becomes_disparity() -> None:
    width, height = 6, 4
    grid = pixel_grid(width, height)
    predicted = grid.copy()
    predicted[:, 0] -= 2.0
    match_valid = np.ones(len(grid), dtype=bool)
    disparity, row_deviation = correspondences_to_disparity(
        grid, predicted, match_valid, width, height, 16.0, 2.0
    )
    assert disparity.shape == (height, width)
    np.testing.assert_allclose(disparity, 2.0)
    assert row_deviation == 0.0


def test_filters_row_deviation_range_and_invalid() -> None:
    width, height = 4, 1
    grid = pixel_grid(width, height)
    predicted = grid.copy()
    predicted[:, 0] -= 3.0
    predicted[1, 1] += 5.0
    predicted[2, 0] = grid[2, 0] + 1.0
    match_valid = np.array([True, True, True, False])
    disparity, _ = correspondences_to_disparity(
        grid, predicted, match_valid, width, height, 16.0, 2.0
    )
    assert np.isfinite(disparity[0, 0])
    assert np.isnan(disparity[0, 1])
    assert np.isnan(disparity[0, 2])
    assert np.isnan(disparity[0, 3])


def test_disparity_above_maximum_is_dropped() -> None:
    width, height = 2, 1
    grid = pixel_grid(width, height)
    predicted = grid.copy()
    predicted[:, 0] -= 20.0
    match_valid = np.ones(len(grid), dtype=bool)
    disparity, _ = correspondences_to_disparity(
        grid, predicted, match_valid, width, height, 16.0, 2.0
    )
    assert np.isnan(disparity).all()
