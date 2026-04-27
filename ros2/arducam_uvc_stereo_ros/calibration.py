"""Calibration parsing and ROS CameraInfo conversion."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
from sensor_msgs.msg import CameraInfo


class CalibrationError(RuntimeError):
    """Raised when calibration data is invalid or incompatible."""


@dataclass(frozen=True)
class CameraCalibration:
    """Calibration for one monocular camera."""

    name: str
    width: int
    height: int
    k: np.ndarray
    raw_d: np.ndarray
    publish_d: np.ndarray
    distortion_model: str


@dataclass(frozen=True)
class StereoCalibration:
    """Parsed stereo calibration payload."""

    version: int
    left: CameraCalibration
    right: CameraCalibration
    rotation: np.ndarray
    translation_m: np.ndarray

    @property
    def image_size(self) -> Tuple[int, int]:
        return self.left.width, self.left.height

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.translation_m))


@dataclass(frozen=True)
class CameraInfoPair:
    """Pair of prebuilt CameraInfo templates."""

    left: CameraInfo
    right: CameraInfo
    width: int
    height: int


def parse_stereo_calibration(
    json_text: str,
    version: int,
    translation_scale_to_meter: float = 0.01,
) -> StereoCalibration:
    """Parse stereo calibration JSON read from device flash."""

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"Calibration JSON is invalid: {exc}") from exc

    camera_data = payload.get("cameraData")
    if not isinstance(camera_data, list):
        raise CalibrationError("Calibration JSON must contain a cameraData list.")

    cameras = _index_camera_entries(camera_data)
    if "left" not in cameras or "right" not in cameras:
        raise CalibrationError("Calibration JSON must contain exactly one left and one right camera entry.")

    left = _parse_camera_entry("left", cameras["left"])
    right = _parse_camera_entry("right", cameras["right"])

    if left.width != right.width or left.height != right.height:
        raise CalibrationError("Left and right calibration dimensions must match.")

    extrinsics = cameras["left"].get("extrinsics")
    if not isinstance(extrinsics, dict):
        raise CalibrationError("Left camera calibration must contain extrinsics to the right camera.")

    to_cam = str(extrinsics.get("to_cam", "")).lower()
    if to_cam != "right":
        raise CalibrationError("Left camera extrinsics must point to the right camera.")

    rotation = _parse_matrix(extrinsics.get("rotationMatrix"), (3, 3), "extrinsics.rotationMatrix")
    translation = _parse_vector(extrinsics.get("translation"), "extrinsics.translation")
    translation_m = translation.reshape(3, 1) * float(translation_scale_to_meter)

    return StereoCalibration(
        version=int(version),
        left=left,
        right=right,
        rotation=rotation,
        translation_m=translation_m,
    )


def build_camera_info_pair(
    calibration: StereoCalibration,
    actual_width: int,
    actual_height: int,
    left_frame_id: str,
    right_frame_id: str,
) -> CameraInfoPair:
    """Build CameraInfo templates for a concrete output size."""

    if actual_width <= 0 or actual_height <= 0:
        raise CalibrationError("Output image size must be positive.")

    calib_width, calib_height = calibration.image_size
    scale_x = float(actual_width) / float(calib_width)
    scale_y = float(actual_height) / float(calib_height)
    if not np.isclose(scale_x, scale_y, rtol=1e-6, atol=1e-9):
        raise CalibrationError(
            f"Output size {actual_width}x{actual_height} changes aspect ratio relative to calibration "
            f"{calib_width}x{calib_height}."
        )

    left_k = _scale_intrinsic_matrix(calibration.left.k, scale_x, scale_y)
    right_k = _scale_intrinsic_matrix(calibration.right.k, scale_x, scale_y)

    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        cameraMatrix1=left_k,
        distCoeffs1=calibration.left.raw_d.reshape(1, -1),
        cameraMatrix2=right_k,
        distCoeffs2=calibration.right.raw_d.reshape(1, -1),
        imageSize=(int(actual_width), int(actual_height)),
        R=calibration.rotation,
        T=calibration.translation_m,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )

    left_info = CameraInfo()
    left_info.header.frame_id = left_frame_id
    left_info.width = int(actual_width)
    left_info.height = int(actual_height)
    left_info.distortion_model = calibration.left.distortion_model
    left_info.d = calibration.left.publish_d.tolist()
    left_info.k = left_k.reshape(-1).tolist()
    left_info.r = np.asarray(r1, dtype=np.float64).reshape(-1).tolist()
    left_info.p = np.asarray(p1, dtype=np.float64).reshape(-1).tolist()

    right_info = CameraInfo()
    right_info.header.frame_id = right_frame_id
    right_info.width = int(actual_width)
    right_info.height = int(actual_height)
    right_info.distortion_model = calibration.right.distortion_model
    right_info.d = calibration.right.publish_d.tolist()
    right_info.k = right_k.reshape(-1).tolist()
    right_info.r = np.asarray(r2, dtype=np.float64).reshape(-1).tolist()
    right_info.p = np.asarray(p2, dtype=np.float64).reshape(-1).tolist()

    return CameraInfoPair(left=left_info, right=right_info, width=int(actual_width), height=int(actual_height))


def clone_camera_info(template: CameraInfo, stamp) -> CameraInfo:
    """Clone a CameraInfo template and apply a new timestamp."""

    message = copy.deepcopy(template)
    message.header.stamp = stamp
    return message


def _index_camera_entries(camera_data: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    indexed: Dict[str, Dict[str, object]] = {}
    for entry in camera_data:
        name = str(entry.get("name", "")).lower()
        if not name:
            raise CalibrationError("Every camera entry must include a name.")
        if name in indexed:
            raise CalibrationError(f"Duplicate camera entry found for {name}.")
        indexed[name] = entry
    return indexed


def _parse_camera_entry(name: str, entry: Dict[str, object]) -> CameraCalibration:
    width = int(entry.get("width", 0))
    height = int(entry.get("height", 0))
    if width <= 0 or height <= 0:
        raise CalibrationError(f"{name} camera width/height must be positive.")

    k = _parse_matrix(entry.get("intrinsicMatrix"), (3, 3), f"{name}.intrinsicMatrix")
    raw_d = _parse_distortion(entry.get("dist_coeff"), f"{name}.dist_coeff")
    publish_d, distortion_model = _publish_distortion(raw_d)
    return CameraCalibration(
        name=name,
        width=width,
        height=height,
        k=k,
        raw_d=raw_d,
        publish_d=publish_d,
        distortion_model=distortion_model,
    )


def _parse_matrix(value: object, shape: Tuple[int, int], field_name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape:
        raise CalibrationError(f"{field_name} must have shape {shape}, got {matrix.shape}.")
    return matrix


def _parse_vector(value: object, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,):
        raise CalibrationError(f"{field_name} must contain exactly 3 values.")
    return vector


def _parse_distortion(value: object, field_name: str) -> np.ndarray:
    distortion = np.asarray(value, dtype=np.float64).reshape(-1)
    if distortion.size < 5:
        raise CalibrationError(f"{field_name} must contain at least 5 values.")
    return distortion


def _publish_distortion(raw_distortion: np.ndarray) -> Tuple[np.ndarray, str]:
    if raw_distortion.size >= 8:
        return raw_distortion[:8], "rational_polynomial"
    return raw_distortion[:5], "plumb_bob"


def _scale_intrinsic_matrix(matrix: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    scaled = np.asarray(matrix, dtype=np.float64).copy()
    scaled[0, 0] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[0, 2] *= scale_x
    scaled[1, 2] *= scale_y
    return scaled
