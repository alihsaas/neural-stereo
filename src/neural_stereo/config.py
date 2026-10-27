from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SCENE_ROOT = DATA_DIR / "MiddEval3" / "trainingQ"

SCENES = ("Motorcycle", "Piano", "Pipes")
JPEG_QUALITIES = (95, 80, 60, 40, 20)
BASELINE_CONDITION = "png"
PAIRS_PER_SECOND = 10

DATA_ZIP_URL = "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip"
GT_ZIP_URL = "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip"

SCENE_FILES = ("im0.png", "im1.png", "calib.txt", "disp0GT.pfm", "mask0nocc.png")


@dataclass(frozen=True)
class SgbmSettings:
    block_size: int = 5
    uniqueness_ratio: int = 10
    disp12_max_diff: int = 1
    speckle_window_size: int = 100
    speckle_range: int = 2

    @property
    def p1(self) -> int:
        return 8 * 3 * self.block_size**2

    @property
    def p2(self) -> int:
        return 32 * 3 * self.block_size**2


def condition_labels() -> tuple[str, ...]:
    return (BASELINE_CONDITION, *(f"q{quality}" for quality in JPEG_QUALITIES))
