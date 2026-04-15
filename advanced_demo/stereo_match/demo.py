#!/usr/bin/env python3
"""
PyQt6 GUI for the Arducam UVC Stereo disparity demo.
"""

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
    make_args,
    prepare_runtime,
    process_disparity_frame,
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
)

NON_RUNTIME_SETTINGS = {"display_mode"}


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

                if runtime is None or signature != runtime_signature:
                    try:
                        runtime = prepare_runtime(
                            params["img_size"],
                            rectification["roi_overlap"],
                            settings,
                            device_index=self.scan_index,
                        )
                        runtime_signature = signature
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
                )
                display_frame = frame_data["display_frame"]

                self.frame_ready.emit(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

            if self._should_stop():
                self.status.emit("Preview stopped")
        except Exception as exc:
            if self._should_stop():
                self.status.emit("Preview stopped")
            else:
                self.error.emit(str(exc))
        finally:
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

        layout.addWidget(QLabel("Mode"))
        layout.addWidget(self.view_mode_combo, stretch=1)
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
