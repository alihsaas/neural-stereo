from neural_stereo.config import RESULTS_DIR, SCENES, SgbmSettings
from neural_stereo.experiments import run_scene
from neural_stereo.matching import sgbm_disparity_factory
from neural_stereo.plotting import plot_payload_vs_error
from neural_stereo.reporting import StudyRow, write_csv


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    settings = SgbmSettings()
    all_rows: list[StudyRow] = []
    for scene in SCENES:
        print(f"scene {scene}")
        all_rows.extend(
            run_scene(
                scene,
                lambda calib: sgbm_disparity_factory(calib, settings),
                RESULTS_DIR / f"{scene.lower()}_grid.png",
            )
        )
    write_csv(all_rows, RESULTS_DIR / "results.csv")
    plot_payload_vs_error(all_rows, RESULTS_DIR / "payload_vs_error.png")
    print(f"wrote CSV and figures to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
