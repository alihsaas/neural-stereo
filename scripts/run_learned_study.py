import functools

import torch

from neural_stereo.calib import Calibration
from neural_stereo.config import RESULTS_DIR, SCENES
from neural_stereo.experiments import run_scene
from neural_stereo.learned import FlexGlueDisparityEstimator, flexglue_disparity_factory
from neural_stereo.plotting import plot_payload_vs_error, plot_payload_vs_error_comparison
from neural_stereo.reporting import StudyRow, read_csv, write_csv


def reporting_factory(estimator: FlexGlueDisparityEstimator, calib: Calibration):
    inner = flexglue_disparity_factory(estimator, calib)

    def estimate(left_bgr, right_bgr):
        disparity = inner(left_bgr, right_bgr)
        diagnostics = estimator.last_diagnostics
        print(
            f"    covisibility {diagnostics['covisibility_score']:.3f}, "
            f"mean abs row deviation {diagnostics['mean_abs_row_deviation_px']:.2f} px"
        )
        return disparity

    return estimate


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")
    estimator = FlexGlueDisparityEstimator(device)
    all_rows: list[StudyRow] = []
    for scene in SCENES:
        print(f"scene {scene}")
        all_rows.extend(
            run_scene(
                scene,
                functools.partial(reporting_factory, estimator),
                RESULTS_DIR / f"{scene.lower()}_grid_flexglue.png",
            )
        )
    write_csv(all_rows, RESULTS_DIR / "results_flexglue.csv")
    plot_payload_vs_error(all_rows, RESULTS_DIR / "payload_vs_error_flexglue.png")
    sgbm_csv = RESULTS_DIR / "results.csv"
    if sgbm_csv.is_file():
        plot_payload_vs_error_comparison(
            [("SGBM", read_csv(sgbm_csv)), ("FlexGlue", all_rows)],
            RESULTS_DIR / "payload_vs_error_comparison.png",
        )
    print(f"wrote CSV and figures to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
