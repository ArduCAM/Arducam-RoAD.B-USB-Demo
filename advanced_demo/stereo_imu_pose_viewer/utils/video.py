from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

DEFAULT_FRAME_SIZE = (2560, 800)


def frame_to_qimage(frame_rgb: Any) -> Any:
    from PyQt6.QtGui import QImage

    frame_rgb = np.ascontiguousarray(frame_rgb)
    height, width = frame_rgb.shape[:2]
    bytes_per_line = frame_rgb.strides[0]
    return QImage(
        frame_rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    ).copy()


def open_camera(
    candidates: list[dict[str, Any]],
    frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
) -> tuple[Any, str, tuple[int, int]]:
    if not candidates:
        raise RuntimeError("no usable capture source")

    errors = []
    for candidate in candidates:
        label = str(candidate["label"])
        time.sleep(0.1)
        backend_id = candidate.get("backend_id")
        if backend_id is None:
            cap = cv2.VideoCapture(candidate["source"])
        else:
            cap = cv2.VideoCapture(candidate["source"], int(backend_id))

        if not cap.isOpened():
            cap.release()
            errors.append(f"{label}: open failed")
            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        width, height = frame_size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        expected_w, expected_h = frame_size
        if actual_w != expected_w or actual_h != expected_h:
            cap.release()
            errors.append(f"{label}: requested {expected_w}x{expected_h}, got {actual_w}x{actual_h}")
            continue

        return cap, label, (actual_w, actual_h)

    raise RuntimeError("cannot open camera from candidates: " + ", ".join(errors))


def validate_stereo_frame(frame: Any, frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE) -> None:
    if frame is None or frame.size == 0:
        raise ValueError("empty frame")

    frame_h, frame_w = frame.shape[:2]
    expected_w, expected_h = frame_size
    if frame_w != expected_w or frame_h != expected_h:
        raise ValueError(f"invalid stereo frame size: expected {expected_w}x{expected_h}, got {frame_w}x{frame_h}")
