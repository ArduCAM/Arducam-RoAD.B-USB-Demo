"""Video capture backend and stereo frame splitting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import cv2


class CaptureError(RuntimeError):
    """Raised when a camera stream cannot be opened or read."""


class FrameFormatError(RuntimeError):
    """Raised when a captured frame is not a valid side-by-side stereo image."""


@dataclass(frozen=True)
class CaptureSettings:
    """Actual capture settings reported by OpenCV."""

    width: int
    height: int
    fps: float
    pixel_format: str


CaptureSource = Union[int, str]


class VideoCaptureBackend:
    """Small wrapper around cv2.VideoCapture using the V4L2 backend."""

    SUPPORTED_PIXEL_FORMATS = {"MJPG", "YUYV"}

    def __init__(
        self,
        source: CaptureSource,
        combined_width: int,
        combined_height: int,
        fps: float,
        pixel_format: str,
        source_description: str = "",
    ) -> None:
        self.source = source
        self.combined_width = int(combined_width)
        self.combined_height = int(combined_height)
        self.fps = float(fps)
        self.pixel_format = pixel_format.upper()
        self.source_description = source_description or describe_capture_source(source)
        self._capture = None

    def open(self) -> CaptureSettings:
        if self.pixel_format not in self.SUPPORTED_PIXEL_FORMATS:
            raise ValueError(
                f"Unsupported pixel_format {self.pixel_format!r}. Expected one of {sorted(self.SUPPORTED_PIXEL_FORMATS)}."
            )

        capture = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
        if not capture or not capture.isOpened():
            raise CaptureError(f"Failed to open UVC capture source {self.source_description}.")

        self._capture = capture

        if self.combined_width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.combined_width)
        if self.combined_height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.combined_height)
        if self.fps > 0.0:
            capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.pixel_format))

        return self.describe()

    def is_opened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def describe(self) -> CaptureSettings:
        if self._capture is None:
            raise CaptureError("Capture is not open.")
        return CaptureSettings(
            width=int(round(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            height=int(round(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            fps=float(self._capture.get(cv2.CAP_PROP_FPS)),
            pixel_format=fourcc_to_string(int(self._capture.get(cv2.CAP_PROP_FOURCC))),
        )

    def read(self):
        if self._capture is None:
            raise CaptureError("Capture is not open.")
        return self._capture.read()

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def describe_capture_source(source: CaptureSource) -> str:
    """Return a human-readable capture source description."""

    if isinstance(source, int):
        return f"OpenCV index {source}"
    return str(source)


def fourcc_to_string(fourcc_value: int) -> str:
    """Convert an OpenCV FOURCC integer into a readable four-character string."""

    chars = [chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)]
    return "".join(chars).rstrip("\x00") or "----"


def split_stereo_frame(frame):
    """Split a side-by-side stereo frame into left/right images and detect encoding."""

    if frame is None:
        raise FrameFormatError("Capture returned an empty frame.")
    if frame.ndim not in (2, 3):
        raise FrameFormatError(f"Unsupported frame rank {frame.ndim}; expected mono or BGR image.")

    width = int(frame.shape[1])
    if width % 2 != 0:
        raise FrameFormatError(f"Combined stereo frame width must be even, got {width}.")

    half_width = width // 2
    if frame.ndim == 2:
        return frame[:, :half_width].copy(), frame[:, half_width:].copy(), "mono8"

    channels = int(frame.shape[2])
    if channels != 3:
        raise FrameFormatError(f"Unsupported channel count {channels}; only mono8 and bgr8 are supported.")
    return frame[:, :half_width, :].copy(), frame[:, half_width:, :].copy(), "bgr8"
