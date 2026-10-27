import re
from dataclasses import dataclass
from pathlib import Path

NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class Calibration:
    focal_length_px: float
    cx0: float
    cy0: float
    doffs: float
    baseline_mm: float
    width: int
    height: int
    ndisp: int


def parse_calib(text: str) -> Calibration:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    cam0_numbers = [float(value) for value in NUMBER_PATTERN.findall(fields["cam0"])]
    return Calibration(
        focal_length_px=cam0_numbers[0],
        cx0=cam0_numbers[2],
        cy0=cam0_numbers[5],
        doffs=float(fields["doffs"]),
        baseline_mm=float(fields["baseline"]),
        width=int(fields["width"]),
        height=int(fields["height"]),
        ndisp=int(fields["ndisp"]),
    )


def load_calib(path: Path) -> Calibration:
    return parse_calib(path.read_text(encoding="ascii"))
