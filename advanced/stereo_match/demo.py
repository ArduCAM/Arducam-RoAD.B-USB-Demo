#!/usr/bin/env python3
"""
PyQt6 GUI for the Arducam UVC Stereo disparity demo.
"""

import os
import sys
import threading
import signal
from pathlib import Path

import cv2
import numpy as np

# Preload the native SDK before PyQt6 updates PATH/Qt DLL search state.
try:
    import arducam_uvc_stereo_sdk as _arducam_sdk_preload  # noqa: F401
except ImportError:
    _arducam_sdk_preload = None

try:
    from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    print(
        "[ERROR] PyQt6 is not installed. Run: python -m pip install -r python/requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from stereo_match.core import (
    DEFAULT_SETTINGS,
    crop_to_roi,
    make_args,
    prepare_runtime,
    process_disparity_frame,
    right_matcher_min_disparity,
    roi_size,
    validate_args,
)

from utils import (
    build_device_entries,
    compute_rectification,
    extract_stereo_params,
    get_capture_candidates,
    open_camera,
    read_device_calibration,
    split_stereo_frame,
)


YOLO_MODEL_PATH = Path(__file__).resolve().parent / "model" / "yolo26n.pt"
YOLO_CONFIG_DIR = Path("/tmp/Ultralytics")
YOLO_CONFIDENCE = 0.5
YOLO_IMAGE_SIZE = 640
YOLO_MAX_DETECTIONS = 100
NON_RUNTIME_SETTINGS = {"display_mode"}
DEPTH_UNIT_LABEL = "cm"
DEPTH_SAMPLE_INSET = 0.2
DEPTH_MIN_VALID_PIXELS = 9


def frame_to_qimage(frame_rgb):
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


def clip_box(box, width, height):
    x1, y1, x2, y2 = (float(value) for value in box)
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return x1, y1, x2, y2


def class_color(class_id):
    return (
        48 + (class_id * 67) % 192,
        48 + (class_id * 131) % 192,
        48 + (class_id * 197) % 192,
    )


def draw_detections(frame_bgr, detections):
    if not detections:
        return frame_bgr

    annotated = frame_bgr.copy()
    height, width = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for detection in detections:
        clipped = clip_box(detection["box"], width, height)
        if clipped is None:
            continue

        x1 = min(max(int(np.floor(clipped[0])), 0), max(width - 2, 0))
        y1 = min(max(int(np.floor(clipped[1])), 0), max(height - 2, 0))
        x2 = min(max(int(np.ceil(clipped[2])), x1 + 1), width - 1)
        y2 = min(max(int(np.ceil(clipped[3])), y1 + 1), height - 1)

        color = detection["color"]
        lines = detection.get("text_lines")
        if not lines:
            lines = [f"{detection['label']} {detection['confidence']:.2f}"]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        metrics = [cv2.getTextSize(line, font, 0.55, 1) for line in lines]
        text_width = max(size[0] for size, _baseline in metrics)
        line_height = max(size[1] + baseline for size, baseline in metrics)
        text_block_height = line_height * len(lines) + 6
        label_top = y1 - text_block_height - 6
        if label_top < 0:
            label_top = min(y1 + 2, max(height - text_block_height - 2, 0))
        label_bottom = min(height - 1, label_top + text_block_height)
        label_right = min(width - 1, x1 + text_width + 8)

        cv2.rectangle(
            annotated,
            (x1, label_top),
            (label_right, label_bottom),
            color,
            -1,
            cv2.LINE_AA,
        )

        text_y = label_top + 2
        for line, (size, baseline) in zip(lines, metrics):
            text_y += size[1] + baseline
            cv2.putText(
                annotated,
                line,
                (x1 + 4, text_y - baseline),
                font,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            text_y += 2

    return annotated


def build_disparity_detection_source(frame, img_size, maps, runtime):
    map_l1, map_l2, _map_r1, _map_r2 = maps
    process_size = runtime["process_size"]
    overlap_roi_proc = runtime["display_config"]["overlap_roi_proc"]

    left, _right = split_stereo_frame(frame, img_size)
    left_rect = cv2.remap(left, map_l1, map_l2, cv2.INTER_LINEAR)
    if process_size != img_size:
        left_rect = cv2.resize(left_rect, process_size, interpolation=cv2.INTER_AREA)
    return crop_to_roi(left_rect, overlap_roi_proc)


def project_detections_to_disparity(detections, disparity_roi_proc, output_size):
    roi_x, roi_y, roi_width, roi_height = disparity_roi_proc
    output_width, output_height = output_size
    if roi_width <= 0 or roi_height <= 0 or output_width <= 0 or output_height <= 0:
        return []

    scale_x = output_width / float(roi_width)
    scale_y = output_height / float(roi_height)

    projected = []
    for detection in detections:
        box = detection["box"]
        mapped_box = (
            (box[0] - roi_x) * scale_x,
            (box[1] - roi_y) * scale_y,
            (box[2] - roi_x) * scale_x,
            (box[3] - roi_y) * scale_y,
        )
        clipped = clip_box(mapped_box, output_width, output_height)
        if clipped is None:
            continue

        projected.append(
            {
                **detection,
                "box": clipped,
            }
        )

    return projected


def resolve_class_name(names, class_id):
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def box_center(box):
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def scale_points(points, src_size, dst_size):
    src_width, src_height = src_size
    dst_width, dst_height = dst_size
    scaled = np.asarray(points, dtype=np.float32).copy()
    scaled[:, 0] *= dst_width / float(src_width)
    scaled[:, 1] *= dst_height / float(src_height)
    return scaled


def points_to_box(points):
    xs = points[:, 0]
    ys = points[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def shrink_box(box, inset_ratio):
    x1, y1, x2, y2 = box
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    inset_x = min(width * inset_ratio, width * 0.45)
    inset_y = min(height * inset_ratio, height * 0.45)
    shrunk = (x1 + inset_x, y1 + inset_y, x2 - inset_x, y2 - inset_y)
    if shrunk[2] - shrunk[0] < 1.0 or shrunk[3] - shrunk[1] < 1.0:
        return box
    return shrunk


def crop_box_to_roi_space(box, roi):
    roi_x, roi_y, roi_width, roi_height = roi
    return clip_box(
        (
            box[0] - roi_x,
            box[1] - roi_y,
            box[2] - roi_x,
            box[3] - roi_y,
        ),
        roi_width,
        roi_height,
    )


def rectify_points(points, camera_name, params, rectification):
    if camera_name == "left":
        camera_matrix = params["K_l"]
        dist_coeffs = params["D_l"]
        rectify_rotation = rectification["R1"]
        projection = rectification["P1"]
    elif camera_name == "right":
        camera_matrix = params["K_r"]
        dist_coeffs = params["D_r"]
        rectify_rotation = rectification["R2"]
        projection = rectification["P2"]
    else:
        raise ValueError(f"unsupported camera name: {camera_name}")

    raw_points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    rectified = cv2.undistortPoints(
        raw_points,
        camera_matrix,
        dist_coeffs,
        R=rectify_rotation,
        P=projection,
    )
    return rectified.reshape(-1, 2)


def raw_box_to_depth_geometry(
    box,
    camera_name,
    params,
    rectification,
    process_size,
    overlap_roi_proc,
):
    img_size = params["img_size"]
    corners = np.asarray(
        [
            [box[0], box[1]],
            [box[2], box[1]],
            [box[2], box[3]],
            [box[0], box[3]],
        ],
        dtype=np.float32,
    )
    rectified_corners = rectify_points(corners, camera_name, params, rectification)
    process_corners = scale_points(rectified_corners, img_size, process_size)
    depth_box = crop_box_to_roi_space(points_to_box(process_corners), overlap_roi_proc)
    if depth_box is None:
        return None

    center = np.asarray([box_center(box)], dtype=np.float32)
    rectified_center = rectify_points(center, camera_name, params, rectification)
    process_center = scale_points(rectified_center, img_size, process_size)[0]
    return {
        "depth_box": depth_box,
        "center_proc": (float(process_center[0]), float(process_center[1])),
    }


def sample_valid_disparity(disparity_map, box, valid_fn, value_fn):
    if disparity_map is None:
        return None

    height, width = disparity_map.shape[:2]
    for candidate_box in (shrink_box(box, DEPTH_SAMPLE_INSET), box):
        clipped = clip_box(candidate_box, width, height)
        if clipped is None:
            continue

        x1 = int(np.floor(clipped[0]))
        y1 = int(np.floor(clipped[1]))
        x2 = int(np.ceil(clipped[2]))
        y2 = int(np.ceil(clipped[3]))
        region = disparity_map[y1:y2, x1:x2]
        if region.size == 0:
            continue

        valid_values = region[valid_fn(region)]
        if valid_values.size >= DEPTH_MIN_VALID_PIXELS or (
            candidate_box == box and valid_values.size > 0
        ):
            transformed = value_fn(valid_values)
            if transformed.size == 0:
                continue
            return float(np.median(transformed))

    return None


def process_intrinsics(rectification, img_size, process_size):
    scale_x = process_size[0] / float(img_size[0])
    scale_y = process_size[1] / float(img_size[1])
    projection_left = rectification["P1"]
    projection_right = rectification["P2"]
    baseline = abs(float(projection_right[0, 3]) / float(projection_right[0, 0]))
    return {
        "fx": float(projection_left[0, 0]) * scale_x,
        "fy": float(projection_left[1, 1]) * scale_y,
        "cx": float(projection_left[0, 2]) * scale_x,
        "cy": float(projection_left[1, 2]) * scale_y,
        "baseline": baseline,
    }


def compute_detection_xyz(detection, frame_data, params, rectification, runtime):
    disparity_map = frame_data["disparity"]
    if disparity_map is None:
        return None

    args = runtime["args"]
    process_size = frame_data["process_size"]
    overlap_roi_proc = runtime["display_config"]["overlap_roi_proc"]
    intrinsics = process_intrinsics(rectification, params["img_size"], process_size)
    if intrinsics["fx"] <= 0.0 or intrinsics["fy"] <= 0.0 or intrinsics["baseline"] <= 0.0:
        return None

    depth_space = detection.get("depth_space")
    if depth_space == "left_proc":
        depth_box = detection.get("depth_box")
        if depth_box is None:
            return None
        disparity_value = sample_valid_disparity(
            disparity_map,
            depth_box,
            lambda region: region > (args.min_disparity - 1.0),
            lambda values: values,
        )
        if disparity_value is None or disparity_value <= 0.0:
            return None
        center_crop = box_center(depth_box)
        center_proc = (
            overlap_roi_proc[0] + center_crop[0],
            overlap_roi_proc[1] + center_crop[1],
        )
        x_left_proc = center_proc[0]
        y_proc = center_proc[1]
    elif depth_space == "left_raw":
        geometry = raw_box_to_depth_geometry(
            detection["depth_box"],
            "left",
            params,
            rectification,
            process_size,
            overlap_roi_proc,
        )
        if geometry is None:
            return None
        disparity_value = sample_valid_disparity(
            disparity_map,
            geometry["depth_box"],
            lambda region: region > (args.min_disparity - 1.0),
            lambda values: values,
        )
        if disparity_value is None or disparity_value <= 0.0:
            return None
        x_left_proc = geometry["center_proc"][0]
        y_proc = geometry["center_proc"][1]
    elif depth_space == "right_raw":
        disparity_right = frame_data["disparity_right"]
        if disparity_right is None:
            return None
        geometry = raw_box_to_depth_geometry(
            detection["depth_box"],
            "right",
            params,
            rectification,
            process_size,
            overlap_roi_proc,
        )
        if geometry is None:
            return None
        disparity_value = sample_valid_disparity(
            disparity_right,
            geometry["depth_box"],
            lambda region: region > (right_matcher_min_disparity(args) - 1.0),
            lambda values: -values,
        )
        if disparity_value is None or disparity_value <= 0.0:
            return None
        x_left_proc = geometry["center_proc"][0] + disparity_value
        y_proc = geometry["center_proc"][1]
    else:
        return None

    z_value = intrinsics["fx"] * intrinsics["baseline"] / disparity_value
    x_value = (x_left_proc - intrinsics["cx"]) * z_value / intrinsics["fx"]
    y_value = (y_proc - intrinsics["cy"]) * z_value / intrinsics["fy"]
    return x_value, y_value, z_value


def attach_xyz_to_detections(detections, frame_data, params, rectification, runtime):
    if not detections:
        return detections

    for detection in detections:
        xyz = compute_detection_xyz(detection, frame_data, params, rectification, runtime)
        detection["xyz"] = xyz
        detection["text_lines"] = [f"{detection['label']} {detection['confidence']:.2f}"]
        if xyz is None:
            detection["text_lines"].extend(["X=n/a", "Y=n/a", "Z=n/a"])
            continue

        x_value, y_value, z_value = xyz
        detection["text_lines"].extend(
            [
                f"X={x_value:.1f}{DEPTH_UNIT_LABEL}",
                f"Y={y_value:.1f}{DEPTH_UNIT_LABEL}",
                f"Z={z_value:.1f}{DEPTH_UNIT_LABEL}",
            ]
        )
    return detections


class AsyncYoloDetector:
    def __init__(self, status_callback):
        self._status_callback = status_callback
        self._thread = None
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._stop_requested = False
        self._pending_request = None
        self._result_cache = {}
        self._latest_error = None
        self._yolo_model = None
        self._generation = 0

    def start(self):
        with self._condition:
            if self._thread is not None:
                return
            self._stop_requested = False
            self._thread = threading.Thread(
                target=self._thread_main,
                name="AsyncYoloDetector",
                daemon=True,
            )
            self._thread.start()

    def stop(self):
        with self._condition:
            thread = self._thread
            if thread is None:
                return
            self._stop_requested = True
            self._pending_request = None
            self._condition.notify_all()

        thread.join(timeout=2.0)
        with self._condition:
            if self._thread is thread:
                self._thread = None

    def clear(self):
        with self._condition:
            self._generation += 1
            self._pending_request = None
            self._result_cache = {}
            self._latest_error = None

    def submit(self, request):
        self.start()
        with self._condition:
            request = request.copy()
            request["generation"] = self._generation
            self._pending_request = request
            self._condition.notify()

    def poll_result(self, signature):
        with self._condition:
            if self._latest_error is not None:
                if self._latest_error["generation"] == self._generation:
                    message = self._latest_error["message"]
                    self._latest_error = None
                    raise RuntimeError(message)
                self._latest_error = None

            return self._result_cache.get(signature)

    def _ensure_model(self):
        if self._yolo_model is not None:
            return self._yolo_model

        if not YOLO_MODEL_PATH.is_file():
            raise RuntimeError(f"YOLO model file not found: {YOLO_MODEL_PATH}")

        self._status_callback(f"Loading YOLO model from {YOLO_MODEL_PATH.name}...")
        os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Run: python -m pip install -r python/requirements.txt"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"failed to import ultralytics: {exc}") from exc

        try:
            self._yolo_model = YOLO(str(YOLO_MODEL_PATH))
        except Exception as exc:
            raise RuntimeError(f"failed to load YOLO model: {exc}") from exc

        self._status_callback("YOLO detection ready")
        return self._yolo_model

    def _infer_detections(self, frame_bgr):
        model = self._ensure_model()
        if frame_bgr.size == 0:
            return []

        try:
            results = model.predict(
                source=frame_bgr,
                conf=YOLO_CONFIDENCE,
                imgsz=YOLO_IMAGE_SIZE,
                max_det=YOLO_MAX_DETECTIONS,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"YOLO inference failed: {exc}") from exc

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(result, "names", getattr(model, "names", {}))
        detections = []
        for xyxy, confidence, cls_value in zip(
            boxes.xyxy.tolist(),
            boxes.conf.tolist(),
            boxes.cls.tolist(),
        ):
            box = clip_box(xyxy, frame_bgr.shape[1], frame_bgr.shape[0])
            if box is None:
                continue

            class_id = int(cls_value)
            detections.append(
                {
                    "box": box,
                    "label": resolve_class_name(names, class_id),
                    "confidence": float(confidence),
                    "color": class_color(class_id),
                }
            )

        return detections

    def _process_request(self, request):
        detections = self._infer_detections(request["source_frame"])
        for detection in detections:
            detection["depth_box"] = detection["box"]
            detection["depth_space"] = request["depth_space"]
        if request["project_to_disparity"]:
            detections = project_detections_to_disparity(
                detections,
                request["disparity_roi_proc"],
                request["output_size"],
            )
        return {
            "generation": request["generation"],
            "signature": request["signature"],
            "detections": detections,
        }

    def _thread_main(self):
        while True:
            with self._condition:
                while not self._stop_requested and self._pending_request is None:
                    self._condition.wait()
                if self._stop_requested:
                    return
                request = self._pending_request
                self._pending_request = None

            try:
                result = self._process_request(request)
            except Exception as exc:
                with self._condition:
                    if request["generation"] != self._generation:
                        continue
                    self._latest_error = {
                        "generation": request["generation"],
                        "message": str(exc),
                    }
                    self._result_cache = {}
                continue

            with self._condition:
                if result["generation"] != self._generation:
                    continue
                self._result_cache[result["signature"]] = result["detections"]


def format_device_vid_pid(dev):
    vid = getattr(dev, "vid", None)
    pid = getattr(dev, "pid", None)

    parts = []
    if vid is not None:
        parts.append(f"vid=0x{int(vid):04x}")
    if pid is not None:
        parts.append(f"pid=0x{int(pid):04x}")
    return " ".join(parts) if parts else str(dev)


class ImageView(QLabel):
    def __init__(self):
        super().__init__("Click 'Start Preview' to display the disparity map")
        self._pixmap = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(960, 540)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setStyleSheet(
            "QLabel { background: #111; color: #ddd; border: 1px solid #333; }"
        )

    def set_frame(self, frame_rgb):
        self._pixmap = QPixmap.fromImage(frame_to_qimage(frame_rgb))
        self._update_pixmap()

    def clear_frame(self, text):
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class DisparityWorker(QObject):
    frame_ready = pyqtSignal(object)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, scan_index, settings):
        super().__init__()
        self.scan_index = scan_index
        self._settings = settings.copy()
        self._stop_requested = False
        self._cap = None
        self._lock = threading.Lock()
        self._yolo_enabled = bool(settings.get("yolo_enabled", False))
        self._yolo_detector = None

    def update_settings(self, settings):
        with self._lock:
            self._settings = settings.copy()

    def stop(self):
        cap = None
        with self._lock:
            self._stop_requested = True
            cap = self._cap
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _should_stop(self):
        with self._lock:
            return self._stop_requested

    def _get_settings(self):
        with self._lock:
            return self._settings.copy()

    def _runtime_signature(self, settings):
        return tuple(
            (key, settings[key])
            for key in sorted(settings)
            if key not in NON_RUNTIME_SETTINGS
        )

    def _resolve_device(self, sdk):
        entries = build_device_entries(sdk.scan())
        for entry in entries:
            if entry["scan_index"] != self.scan_index:
                continue
            if not entry["has_capture_source"]:
                raise RuntimeError("selected device does not expose a usable capture source")
            return entry["dev"]
        raise RuntimeError(
            f"device index {self.scan_index} not found in current scan result"
        )

    def _runtime_message(self, runtime):
        args = runtime["args"]
        process_width, process_height = roi_size(
            runtime["display_config"]["overlap_roi_proc"]
        )
        message = (
            f"Preview: disparity={process_width}x{process_height}, "
            f"downscale={args.downscale:.2f}, "
            f"numDisparities={args.num_disparities}, blockSize={args.block_size}"
        )
        message += f", {self._post_filter_message(args)}"
        message += f", {self._confidence_message(args)}"
        if args.hole_fill != "off" and args.hole_fill_radius > 0:
            message += (
                f", holeFill={args.hole_fill}"
                f"(r={args.hole_fill_radius}, eps={args.hole_fill_eps:.6f})"
            )
        else:
            message += ", holeFill=off"
        return message

    def _post_filter_message(self, args):
        mode = getattr(args, "post_filter_mode", "bilateral")
        if mode == "off":
            return "filter=off"
        if mode == "bilateral":
            return (
                f"filter=bilateral(d={args.smooth_diameter}, "
                f"c={args.smooth_sigma_color:.2f}, "
                f"s={args.smooth_sigma_space:.2f})"
            )
        if mode == "median":
            return f"filter=median(k={args.median_ksize})"
        if mode == "gaussian":
            return (
                f"filter=gaussian(k={args.gaussian_ksize}, "
                f"sigma={args.gaussian_sigma:.2f})"
            )
        if mode == "wls":
            return (
                f"filter=wls(lambda={args.wls_lambda:.1f}, "
                f"sigma={args.wls_sigma_color:.2f})"
            )
        return f"filter={mode}"

    def _confidence_message(self, args):
        if getattr(args, "confidence_threshold", 0) <= 0:
            return "confidence=off"
        return f"confidence>={args.confidence_threshold}"

    def _ensure_yolo_detector(self):
        if self._yolo_detector is None:
            self._yolo_detector = AsyncYoloDetector(self.status.emit)
        return self._yolo_detector

    def _clear_yolo_results(self):
        if self._yolo_detector is not None:
            self._yolo_detector.clear()

    def _stop_yolo_detector(self):
        if self._yolo_detector is not None:
            self._yolo_detector.stop()
            self._yolo_detector = None

    def _yolo_signature(self, runtime, display_frame_bgr):
        display_mode = getattr(runtime["args"], "display_mode", "disparity")
        output_size = (display_frame_bgr.shape[1], display_frame_bgr.shape[0])
        if display_mode == "disparity":
            return (
                "disparity",
                output_size,
                tuple(int(value) for value in runtime["process_size"]),
                tuple(int(value) for value in runtime["display_config"]["overlap_roi_proc"]),
                tuple(int(value) for value in runtime["display_config"]["disparity_roi_proc"]),
            )
        return (display_mode, output_size)

    def _submit_yolo_request(self, frame, img_size, maps, runtime, display_frame_bgr):
        detector = self._ensure_yolo_detector()
        display_mode = getattr(runtime["args"], "display_mode", "disparity")
        signature = self._yolo_signature(runtime, display_frame_bgr)

        if display_mode in {"left", "right"}:
            source_frame = display_frame_bgr.copy()
            project_to_disparity = False
            disparity_roi_proc = None
            output_size = None
            depth_space = "left_raw" if display_mode == "left" else "right_raw"
        else:
            source_frame = build_disparity_detection_source(
                frame,
                img_size,
                maps,
                runtime,
            ).copy()
            project_to_disparity = True
            disparity_roi_proc = tuple(
                int(value) for value in runtime["display_config"]["disparity_roi_proc"]
            )
            output_size = (display_frame_bgr.shape[1], display_frame_bgr.shape[0])
            depth_space = "left_proc"

        detector.submit(
            {
                "signature": signature,
                "source_frame": source_frame,
                "project_to_disparity": project_to_disparity,
                "disparity_roi_proc": disparity_roi_proc,
                "output_size": output_size,
                "depth_space": depth_space,
            }
        )
        return detector.poll_result(signature)

    def _apply_display_mode(self, runtime, settings):
        runtime["args"].display_mode = settings.get("display_mode", "disparity")

    @pyqtSlot()
    def run(self):
        cap = None

        try:
            from arducam_uvc_stereo_sdk import UVCStereo

            self.status.emit("Scanning devices...")
            sdk = UVCStereo()
            dev = self._resolve_device(sdk)

            self.status.emit("Reading calibration data...")
            calibration = read_device_calibration(sdk, dev)
            params = extract_stereo_params(calibration)
            rectification = compute_rectification(params)
            maps = rectification["maps"]

            candidates = get_capture_candidates(dev)
            if not candidates:
                raise RuntimeError("selected device does not expose a usable capture source")

            self.status.emit("Opening camera...")
            cap = open_camera(candidates, *params["img_size"])
            with self._lock:
                self._cap = cap

            runtime = None
            runtime_signature = None
            while not self._should_stop():
                settings = self._get_settings()
                signature = self._runtime_signature(settings)
                yolo_enabled = bool(settings.get("yolo_enabled", False))

                if yolo_enabled != self._yolo_enabled:
                    self.status.emit(
                        "YOLO detection enabled" if yolo_enabled else "YOLO detection disabled"
                    )
                    self._yolo_enabled = yolo_enabled
                    self._clear_yolo_results()

                if runtime is None or signature != runtime_signature:
                    try:
                        runtime = prepare_runtime(
                            params["img_size"],
                            rectification["roi_overlap"],
                            settings,
                            device_index=self.scan_index,
                        )
                        runtime_signature = signature
                        self._clear_yolo_results()
                        self.status.emit(self._runtime_message(runtime))
                    except Exception as exc:
                        if runtime is None:
                            raise
                        self.status.emit(
                            f"Invalid parameters, continuing with previous settings: {exc}"
                        )
                self._apply_display_mode(runtime, settings)

                ret, frame = cap.read()
                if self._should_stop():
                    break
                if not ret:
                    if self._should_stop():
                        break
                    raise RuntimeError("failed to read frame from camera")

                frame_data = process_disparity_frame(
                    frame,
                    params["img_size"],
                    maps,
                    runtime,
                    output_rgb=False,
                    return_metadata=True,
                    force_disparity=yolo_enabled,
                )
                display_frame = frame_data["display_frame"]
                if yolo_enabled:
                    detections = self._submit_yolo_request(
                        frame,
                        params["img_size"],
                        maps,
                        runtime,
                        display_frame,
                    )
                    if detections is not None:
                        detections = attach_xyz_to_detections(
                            detections,
                            frame_data,
                            params,
                            rectification,
                            runtime,
                        )
                        display_frame = draw_detections(display_frame, detections)

                self.frame_ready.emit(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

            if self._should_stop():
                self.status.emit("Preview stopped")
        except Exception as exc:
            if self._should_stop():
                self.status.emit("Preview stopped")
            else:
                self.error.emit(str(exc))
        finally:
            self._stop_yolo_detector()
            if cap is not None:
                cap.release()
            with self._lock:
                if self._cap is cap:
                    self._cap = None
            self.finished.emit()


class DisparityMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arducam Stereo Disparity GUI")
        self.resize(1480, 900)

        self.worker = None
        self.worker_thread = None
        self._updating_controls = False
        self._shutdown_requested = False

        self.preview = ImageView()
        self.view_mode_combo = QComboBox()
        self.device_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh Devices")
        self.start_button = QPushButton("Start Preview")
        self.stop_button = QPushButton("Stop")
        self.reset_button = QPushButton("Reset Defaults")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(300)

        self.controls = {}
        self._build_ui()
        self._connect_signals()
        self.refresh_devices()
        self.apply_settings_to_controls(DEFAULT_SETTINGS)

    def _build_ui(self):
        root = QWidget()
        root_layout = QHBoxLayout(root)

        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self._build_view_group(), stretch=0)
        preview_layout.addWidget(self.preview, stretch=1)

        log_group = QGroupBox("Status")
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.log_view)
        preview_layout.addWidget(log_group, stretch=0)

        root_layout.addLayout(preview_layout, stretch=3)

        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.addWidget(self._build_device_group())
        controls_layout.addWidget(self._build_parameter_group())
        controls_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls_container)
        scroll.setMinimumWidth(380)
        root_layout.addWidget(scroll, stretch=1)

        self.setCentralWidget(root)

    def _build_view_group(self):
        group = QGroupBox("Display")
        layout = QHBoxLayout(group)

        self.view_mode_combo.addItem("Disparity", "disparity")
        self.view_mode_combo.addItem("Left Camera", "left")
        self.view_mode_combo.addItem("Right Camera", "right")
        self.controls["yolo_enabled"] = QCheckBox("YOLO Detection")

        layout.addWidget(QLabel("Mode"))
        layout.addWidget(self.view_mode_combo, stretch=1)
        layout.addWidget(self.controls["yolo_enabled"])
        return group

    def _build_device_group(self):
        group = QGroupBox("Device")
        layout = QVBoxLayout(group)

        form = QFormLayout()
        form.addRow("Camera", self.device_combo)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addWidget(self.refresh_button)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        layout.addLayout(buttons)
        layout.addWidget(self.reset_button)

        self.stop_button.setEnabled(False)
        return group

    def _build_parameter_group(self):
        group = QGroupBox("Parameters")
        layout = QFormLayout(group)

        self.controls["downscale"] = self._make_double_spin(0.10, 1.00, 0.05, 2)
        self.controls["min_disparity"] = self._make_spin(-256, 256, 1)
        self.controls["num_disparities"] = self._make_spin(16, 512, 16)
        self.controls["block_size"] = self._make_spin(3, 31, 2)
        self.controls["uniqueness_ratio"] = self._make_spin(0, 100, 1)
        self.controls["speckle_window_size"] = self._make_spin(0, 500, 1)
        self.controls["speckle_range"] = self._make_spin(0, 100, 1)
        self.controls["disp12_max_diff"] = self._make_spin(0, 100, 1)

        post_filter_combo = QComboBox()
        post_filter_combo.addItem("Off", "off")
        post_filter_combo.addItem("Bilateral", "bilateral")
        post_filter_combo.addItem("Median", "median")
        post_filter_combo.addItem("Gaussian", "gaussian")
        post_filter_combo.addItem("WLS", "wls")
        self.controls["post_filter_mode"] = post_filter_combo

        self.controls["smooth_diameter"] = self._make_spin(0, 31, 1)
        self.controls["smooth_sigma_color"] = self._make_double_spin(0.0, 50.0, 0.1, 2)
        self.controls["smooth_sigma_space"] = self._make_double_spin(0.0, 50.0, 0.5, 2)
        self.controls["median_ksize"] = self._make_spin(3, 5, 2)
        self.controls["gaussian_ksize"] = self._make_spin(3, 31, 2)
        self.controls["gaussian_sigma"] = self._make_double_spin(0.0, 20.0, 0.1, 2)
        self.controls["wls_lambda"] = self._make_double_spin(0.0, 50000.0, 100.0, 1)
        self.controls["wls_sigma_color"] = self._make_double_spin(0.0, 10.0, 0.1, 2)
        self.controls["confidence_threshold"] = self._make_spin(0, 100, 1)

        hole_fill_combo = QComboBox()
        hole_fill_combo.addItem("Off", "off")
        hole_fill_combo.addItem("Guided Filter", "guided")
        self.controls["hole_fill"] = hole_fill_combo
        self.controls["hole_fill_radius"] = self._make_spin(0, 31, 1)
        self.controls["hole_fill_eps"] = self._make_double_spin(0.0, 1.0, 0.0005, 6)

        layout.addRow("Downscale", self.controls["downscale"])
        layout.addRow("Min Disparity", self.controls["min_disparity"])
        layout.addRow("Num Disparities", self.controls["num_disparities"])
        layout.addRow("Block Size", self.controls["block_size"])
        layout.addRow("Uniqueness Ratio", self.controls["uniqueness_ratio"])
        layout.addRow("Speckle Window Size", self.controls["speckle_window_size"])
        layout.addRow("Speckle Range", self.controls["speckle_range"])
        layout.addRow("Disp12 Max Diff", self.controls["disp12_max_diff"])
        layout.addRow("Post Filter", self.controls["post_filter_mode"])
        layout.addRow("Bilateral Diameter", self.controls["smooth_diameter"])
        layout.addRow("Bilateral Sigma Color", self.controls["smooth_sigma_color"])
        layout.addRow("Bilateral Sigma Space", self.controls["smooth_sigma_space"])
        layout.addRow("Median Kernel", self.controls["median_ksize"])
        layout.addRow("Gaussian Kernel", self.controls["gaussian_ksize"])
        layout.addRow("Gaussian Sigma", self.controls["gaussian_sigma"])
        layout.addRow("WLS Lambda", self.controls["wls_lambda"])
        layout.addRow("WLS Sigma Color", self.controls["wls_sigma_color"])
        layout.addRow("Confidence Threshold", self.controls["confidence_threshold"])
        layout.addRow("Hole Fill", self.controls["hole_fill"])
        layout.addRow("Hole Fill Radius", self.controls["hole_fill_radius"])
        layout.addRow("Hole Fill Eps", self.controls["hole_fill_eps"])
        return group

    def _connect_signals(self):
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.start_button.clicked.connect(self.start_preview)
        self.stop_button.clicked.connect(self.stop_preview)
        self.reset_button.clicked.connect(self.reset_settings)
        self.view_mode_combo.currentIndexChanged.connect(self.on_settings_changed)

        for control in self.controls.values():
            if isinstance(control, QComboBox):
                control.currentIndexChanged.connect(self.on_settings_changed)
            elif isinstance(control, QCheckBox):
                control.toggled.connect(self.on_settings_changed)
            elif isinstance(control, QSpinBox):
                control.valueChanged.connect(self.on_settings_changed)
            elif isinstance(control, QDoubleSpinBox):
                control.valueChanged.connect(self.on_settings_changed)

    def _make_spin(self, minimum, maximum, step):
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        return widget

    def _make_double_spin(self, minimum, maximum, step, decimals):
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setDecimals(decimals)
        return widget

    def _refresh_parameter_states(self):
        filter_mode = self.controls["post_filter_mode"].currentData()
        for name in ("smooth_diameter", "smooth_sigma_color", "smooth_sigma_space"):
            self.controls[name].setEnabled(filter_mode == "bilateral")
        self.controls["median_ksize"].setEnabled(filter_mode == "median")
        self.controls["gaussian_ksize"].setEnabled(filter_mode == "gaussian")
        self.controls["gaussian_sigma"].setEnabled(filter_mode == "gaussian")
        self.controls["wls_lambda"].setEnabled(filter_mode == "wls")
        self.controls["wls_sigma_color"].setEnabled(filter_mode == "wls")

        hole_fill_enabled = self.controls["hole_fill"].currentData() != "off"
        self.controls["hole_fill_radius"].setEnabled(hole_fill_enabled)
        self.controls["hole_fill_eps"].setEnabled(hole_fill_enabled)

    def _append_log(self, message):
        self.log_view.appendPlainText(message)

    def _set_running_state(self, running):
        self.refresh_button.setEnabled(not running)
        self.start_button.setEnabled(not running and self.device_combo.count() > 0)
        self.stop_button.setEnabled(running)
        self.device_combo.setEnabled(not running)

    def current_settings(self):
        return {
            "display_mode": self.view_mode_combo.currentData(),
            "yolo_enabled": self.controls["yolo_enabled"].isChecked(),
            "downscale": self.controls["downscale"].value(),
            "min_disparity": self.controls["min_disparity"].value(),
            "num_disparities": self.controls["num_disparities"].value(),
            "block_size": self.controls["block_size"].value(),
            "uniqueness_ratio": self.controls["uniqueness_ratio"].value(),
            "speckle_window_size": self.controls["speckle_window_size"].value(),
            "speckle_range": self.controls["speckle_range"].value(),
            "disp12_max_diff": self.controls["disp12_max_diff"].value(),
            "post_filter_mode": self.controls["post_filter_mode"].currentData(),
            "smooth_diameter": self.controls["smooth_diameter"].value(),
            "smooth_sigma_color": self.controls["smooth_sigma_color"].value(),
            "smooth_sigma_space": self.controls["smooth_sigma_space"].value(),
            "median_ksize": self.controls["median_ksize"].value(),
            "gaussian_ksize": self.controls["gaussian_ksize"].value(),
            "gaussian_sigma": self.controls["gaussian_sigma"].value(),
            "wls_lambda": self.controls["wls_lambda"].value(),
            "wls_sigma_color": self.controls["wls_sigma_color"].value(),
            "confidence_threshold": self.controls["confidence_threshold"].value(),
            "hole_fill": self.controls["hole_fill"].currentData(),
            "hole_fill_radius": self.controls["hole_fill_radius"].value(),
            "hole_fill_eps": self.controls["hole_fill_eps"].value(),
        }

    def apply_settings_to_controls(self, settings):
        merged = DEFAULT_SETTINGS.copy()
        merged.update(settings)
        self._updating_controls = True
        try:
            index = self.view_mode_combo.findData(merged["display_mode"])
            if index >= 0:
                self.view_mode_combo.setCurrentIndex(index)
            self.controls["yolo_enabled"].setChecked(bool(merged.get("yolo_enabled", False)))
            self.controls["downscale"].setValue(merged["downscale"])
            self.controls["min_disparity"].setValue(merged["min_disparity"])
            self.controls["num_disparities"].setValue(merged["num_disparities"])
            self.controls["block_size"].setValue(merged["block_size"])
            self.controls["uniqueness_ratio"].setValue(merged["uniqueness_ratio"])
            self.controls["speckle_window_size"].setValue(merged["speckle_window_size"])
            self.controls["speckle_range"].setValue(merged["speckle_range"])
            self.controls["disp12_max_diff"].setValue(merged["disp12_max_diff"])
            self.controls["smooth_diameter"].setValue(merged["smooth_diameter"])
            self.controls["smooth_sigma_color"].setValue(merged["smooth_sigma_color"])
            self.controls["smooth_sigma_space"].setValue(merged["smooth_sigma_space"])
            self.controls["median_ksize"].setValue(merged["median_ksize"])
            self.controls["gaussian_ksize"].setValue(merged["gaussian_ksize"])
            self.controls["gaussian_sigma"].setValue(merged["gaussian_sigma"])
            self.controls["wls_lambda"].setValue(merged["wls_lambda"])
            self.controls["wls_sigma_color"].setValue(merged["wls_sigma_color"])
            self.controls["confidence_threshold"].setValue(merged["confidence_threshold"])
            self.controls["hole_fill_radius"].setValue(merged["hole_fill_radius"])
            self.controls["hole_fill_eps"].setValue(merged["hole_fill_eps"])

            index = self.controls["post_filter_mode"].findData(merged["post_filter_mode"])
            if index >= 0:
                self.controls["post_filter_mode"].setCurrentIndex(index)

            index = self.controls["hole_fill"].findData(merged["hole_fill"])
            if index >= 0:
                self.controls["hole_fill"].setCurrentIndex(index)
        finally:
            self._updating_controls = False
            self._refresh_parameter_states()

    def reset_settings(self):
        self.apply_settings_to_controls(DEFAULT_SETTINGS)
        self.on_settings_changed()
        self._append_log("Parameters reset to defaults")

    def refresh_devices(self):
        try:
            from arducam_uvc_stereo_sdk import UVCStereo

            sdk = UVCStereo()
            entries = build_device_entries(sdk.scan())
        except Exception as exc:
            self.device_combo.clear()
            self._set_running_state(False)
            self._append_log(f"Device refresh failed: {exc}")
            QMessageBox.critical(self, "Device Scan Failed", str(exc))
            return

        previous = self.device_combo.currentData()
        self.device_combo.clear()

        for entry in entries:
            if not entry["has_capture_source"]:
                continue
            label = format_device_vid_pid(entry["dev"])
            self.device_combo.addItem(label, entry["scan_index"])

        if self.device_combo.count() == 0:
            self.preview.clear_frame("No usable stereo devices found")
            self._append_log("No usable stereo devices found")
            self._set_running_state(False)
            self.start_button.setEnabled(False)
            return

        if previous is not None:
            previous_index = self.device_combo.findData(previous)
            if previous_index >= 0:
                self.device_combo.setCurrentIndex(previous_index)

        self._append_log(f"Found {self.device_combo.count()} usable device(s)")
        self._set_running_state(False)

    def start_preview(self):
        if self.worker is not None:
            return

        scan_index = self.device_combo.currentData()
        if scan_index is None:
            QMessageBox.warning(self, "No Device Selected", "Please select a usable device first")
            return

        try:
            validate_args(make_args(self.current_settings(), device_index=scan_index))
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Parameters", str(exc))
            return

        self.preview.clear_frame("Starting preview...")
        self._set_running_state(True)
        self._append_log(f"Starting preview for device[{scan_index}]")

        self.worker_thread = QThread(self)
        self.worker = DisparityWorker(scan_index, self.current_settings())
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.status.connect(self.on_worker_status)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.on_thread_finished)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def stop_preview(self):
        if self.worker is not None:
            self.worker.stop()
            self._append_log("Stopping preview...")
            self.stop_button.setEnabled(False)

    def on_settings_changed(self):
        if self._updating_controls:
            return

        self._refresh_parameter_states()
        settings = self.current_settings()
        if self.worker is not None:
            self.worker.update_settings(settings)

    @pyqtSlot(object)
    def on_frame_ready(self, frame_rgb):
        self.preview.set_frame(frame_rgb)

    @pyqtSlot(str)
    def on_worker_status(self, message):
        self._append_log(message)

    @pyqtSlot(str)
    def on_worker_error(self, message):
        self._append_log(f"Error: {message}")
        QMessageBox.critical(self, "Preview Error", message)

    @pyqtSlot()
    def on_worker_finished(self):
        self.worker = None
        self._set_running_state(False)

    @pyqtSlot()
    def on_thread_finished(self):
        self.worker_thread = None

    def shutdown(self):
        if self._shutdown_requested:
            return

        self._shutdown_requested = True
        self.stop_preview()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(2000)

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)


def install_sigint_handler(app, window):
    def handle_sigint(_signum, _frame):
        print("\n[INFO] Received Ctrl+C, shutting down...", file=sys.stderr)
        QTimer.singleShot(0, window.close)

    heartbeat = QTimer()
    heartbeat.setInterval(200)
    heartbeat.timeout.connect(lambda: None)
    heartbeat.start()

    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handle_sigint)
    return heartbeat, previous_handler


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = DisparityMainWindow()
    app.aboutToQuit.connect(window.shutdown)
    sigint_heartbeat, previous_sigint_handler = install_sigint_handler(app, window)
    app._sigint_heartbeat = sigint_heartbeat
    window.show()
    try:
        return app.exec()
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)


if __name__ == "__main__":
    sys.exit(main())
