from pathlib import Path

import numpy as np


def read_pfm(path: Path) -> np.ndarray:
    with open(path, "rb") as stream:
        header = stream.readline().decode("ascii").strip()
        if header not in ("Pf", "PF"):
            raise ValueError(f"not a PFM file: {path}")
        dims_line = stream.readline().decode("ascii")
        while dims_line.strip().startswith("#"):
            dims_line = stream.readline().decode("ascii")
        width, height = (int(value) for value in dims_line.split())
        scale = float(stream.readline().decode("ascii").strip())
        channels = 3 if header == "PF" else 1
        dtype = "<f4" if scale < 0 else ">f4"
        payload = stream.read(width * height * channels * 4)
        data = np.frombuffer(payload, dtype=dtype)
        shape = (height, width, channels) if channels == 3 else (height, width)
        return np.flipud(data.reshape(shape)).astype(np.float32)
