import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class StudyRow:
    scene: str
    condition: str
    jpeg_quality: int | None
    bytes_pair: int
    mbps_estimate_10fps: float
    coverage_pct: float
    bad2px_pct: float
    median_abs_depth_error_cm: float
    median_abs_depth_error_common_cm: float


def write_csv(rows: list[StudyRow], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    column_names = [field.name for field in fields(StudyRow)]
    with open(destination, "w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=column_names)
        writer.writeheader()
        for row in rows:
            record = asdict(row)
            record["jpeg_quality"] = "" if row.jpeg_quality is None else row.jpeg_quality
            record["mbps_estimate_10fps"] = f"{row.mbps_estimate_10fps:.3f}"
            record["coverage_pct"] = f"{row.coverage_pct:.2f}"
            record["bad2px_pct"] = f"{row.bad2px_pct:.2f}"
            record["median_abs_depth_error_cm"] = f"{row.median_abs_depth_error_cm:.2f}"
            record["median_abs_depth_error_common_cm"] = (
                f"{row.median_abs_depth_error_common_cm:.2f}"
            )
            writer.writerow(record)


def read_csv(source: Path) -> list[StudyRow]:
    with open(source, newline="", encoding="ascii") as stream:
        return [
            StudyRow(
                scene=record["scene"],
                condition=record["condition"],
                jpeg_quality=int(record["jpeg_quality"]) if record["jpeg_quality"] else None,
                bytes_pair=int(record["bytes_pair"]),
                mbps_estimate_10fps=float(record["mbps_estimate_10fps"]),
                coverage_pct=float(record["coverage_pct"]),
                bad2px_pct=float(record["bad2px_pct"]),
                median_abs_depth_error_cm=float(record["median_abs_depth_error_cm"]),
                median_abs_depth_error_common_cm=float(record["median_abs_depth_error_common_cm"]),
            )
            for record in csv.DictReader(stream)
        ]


def mbps_estimate(bytes_pair: int, pairs_per_second: int) -> float:
    return bytes_pair * 8 * pairs_per_second / 1e6
