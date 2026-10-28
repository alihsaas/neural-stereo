import cv2
import numpy as np


def jpeg_roundtrip(image: np.ndarray, quality: int) -> tuple[np.ndarray, int]:
    encoded, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not encoded:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return decoded, int(buffer.size)


def png_encoded_size(image: np.ndarray) -> int:
    encoded, buffer = cv2.imencode(".png", image)
    if not encoded:
        raise RuntimeError("PNG encoding failed")
    return int(buffer.size)
