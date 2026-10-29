from pathlib import Path

from neural_stereo.reporting import StudyRow, read_csv, write_csv


def test_csv_round_trip(tmp_path: Path) -> None:
    rows = [
        StudyRow("Piano", "png", None, 1000, 0.08, 90.0, 10.0, 1.23, 1.2),
        StudyRow("Piano", "q80", 80, 500, 0.04, 88.5, 12.25, 2.0, 1.75),
    ]
    destination = tmp_path / "rows.csv"
    write_csv(rows, destination)
    assert read_csv(destination) == rows
