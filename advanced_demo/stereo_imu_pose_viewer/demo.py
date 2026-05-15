#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    import arducam_uvc_stereo_sdk as _arducam_sdk_preload  # noqa: F401
except ImportError:
    _arducam_sdk_preload = None

try:
    import cv2
    from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
    from PyQt6.QtGui import QPixmap
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
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    print(
        "[ERROR] Missing dependency. Run: python -m pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from utils.devices import (
    build_device_entries,
    format_device,
    get_capture_candidates,
    resolve_device,
)
from utils.imu import close_imu, make_conversion_options, sample_from_sdk
from utils.orientation import OrientationEstimator
from utils.pose_view import Pose3DWidget
from utils.video import (
    DEFAULT_FRAME_SIZE,
    frame_to_qimage,
    open_camera,
    validate_stereo_frame,
)


DEFAULT_INTERVAL_MS = 5.0
DEFAULT_ACCEL_BLEND = 0.02


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def blend_value(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arducam stereo image and IMU pose viewer."
    )
    parser.add_argument(
        "--interval-ms",
        type=positive_float,
        default=DEFAULT_INTERVAL_MS,
        help=f"Delay between IMU reads in milliseconds. Default: {DEFAULT_INTERVAL_MS:g}.",
    )
    parser.add_argument(
        "--accel-blend",
        type=blend_value,
        default=DEFAULT_ACCEL_BLEND,
        help=f"Accelerometer correction blend for roll and pitch. Default: {DEFAULT_ACCEL_BLEND:g}.",
    )
    parser.add_argument(
        "--gyro-only",
        action="store_true",
        help="Disable accelerometer correction and use gyroscope integration only.",
    )
    return parser.parse_args(argv)


class ImageView(QLabel):
    def __init__(self) -> None:
        super().__init__("Click 'Start' to display the stereo camera")
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(960, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "QLabel { background: #111; color: #ddd; border: 1px solid #333; }"
        )

    def set_frame(self, frame_rgb: Any) -> None:
        self._pixmap = QPixmap.fromImage(frame_to_qimage(frame_rgb))
        self.setText("")
        self._update_pixmap()

    def clear_frame(self, text: str) -> None:
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class VideoWorker(QObject):
    frame_ready = pyqtSignal(object)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, scan_index: int) -> None:
        super().__init__()
        self.scan_index = int(scan_index)
        self._stop_event = threading.Event()
        self._cap_lock = threading.Lock()
        self._cap: Any = None

    def stop(self) -> None:
        self._stop_event.set()
        with self._cap_lock:
            cap = self._cap
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _should_stop(self) -> bool:
        return self._stop_event.is_set()

    @pyqtSlot()
    def run(self) -> None:
        cap = None
        try:
            from arducam_uvc_stereo_sdk import scan_devices

            self.status.emit("Scanning selected video device...")
            dev = resolve_device(scan_devices(), self.scan_index)
            candidates = get_capture_candidates(dev)
            if not candidates:
                raise RuntimeError(
                    "selected device does not expose a usable capture source"
                )

            self.status.emit("Opening camera...")
            cap, label, actual_size = open_camera(
                candidates, frame_size=DEFAULT_FRAME_SIZE
            )
            with self._cap_lock:
                self._cap = cap
            self.status.emit(
                f"Camera opened via {label}: {actual_size[0]}x{actual_size[1]}"
            )

            while not self._should_stop():
                ret, frame = cap.read()
                if self._should_stop():
                    break
                if not ret or frame is None:
                    raise RuntimeError("failed to read frame from camera")

                validate_stereo_frame(frame, DEFAULT_FRAME_SIZE)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(frame_rgb)

            if self._should_stop():
                self.status.emit("Video stopped")
        except Exception as exc:
            if self._should_stop():
                self.status.emit("Video stopped")
            else:
                self.error.emit(f"Video error: {exc}")
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            with self._cap_lock:
                if self._cap is cap:
                    self._cap = None
            self.finished.emit()


class ImuWorker(QObject):
    pose_ready = pyqtSignal(object, object, float, float)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self, scan_index: int, interval_ms: float, accel_blend: float, use_accel: bool
    ) -> None:
        super().__init__()
        self.scan_index = int(scan_index)
        self.interval_ms = float(interval_ms)
        self.accel_blend = float(accel_blend)
        self.use_accel = bool(use_accel)
        self._stop_event = threading.Event()
        self._device_lock = threading.Lock()
        self._imu_device: Any = None

    def stop(self) -> None:
        self._stop_event.set()
        with self._device_lock:
            device = self._imu_device
        if device is not None:
            close_imu(device)

    def _should_stop(self) -> bool:
        return self._stop_event.is_set()

    @pyqtSlot()
    def run(self) -> None:
        imu_device = None
        try:
            from arducam_uvc_stereo_sdk import open_device, scan_devices

            self.status.emit("Scanning selected IMU device...")
            dev = resolve_device(scan_devices(), self.scan_index)
            self.status.emit(f"Opening IMU: {format_device(dev)}")
            imu_device = open_device(dev)
            with self._device_lock:
                self._imu_device = imu_device
            imu_device.open_imu()

            conversion_options = make_conversion_options()
            estimator = OrientationEstimator(
                accel_blend=self.accel_blend, use_accel=self.use_accel
            )
            self.status.emit(
                f"IMU opened: interval={self.interval_ms:g} ms, accel={'on' if self.use_accel else 'off'}"
            )

            last_read_done: float | None = None
            while not self._should_stop():
                loop_start = time.monotonic()
                raw = imu_device.read_imu()
                read_done = time.monotonic()

                read_ms = (read_done - loop_start) * 1000.0
                interval_ms = (
                    0.0
                    if last_read_done is None
                    else (read_done - last_read_done) * 1000.0
                )
                last_read_done = read_done

                sample = sample_from_sdk(raw, conversion_options)
                dt_s = interval_ms / 1000.0 if interval_ms > 0.0 else None
                pose = estimator.update(sample, dt_s=dt_s)
                self.pose_ready.emit(pose, sample, read_ms, interval_ms)

                sleep_until = loop_start + self.interval_ms / 1000.0
                while not self._should_stop():
                    remaining = sleep_until - time.monotonic()
                    if remaining <= 0.0:
                        break
                    time.sleep(min(0.005, remaining))

            if self._should_stop():
                self.status.emit("IMU stopped")
        except Exception as exc:
            if self._should_stop():
                self.status.emit("IMU stopped")
            else:
                self.error.emit(f"IMU error: {exc}")
        finally:
            if imu_device is not None:
                close_imu(imu_device)
            with self._device_lock:
                if self._imu_device is imu_device:
                    self._imu_device = None
            self.finished.emit()


class StereoImuMainWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.setWindowTitle("Arducam Stereo IMU Demo")
        self.resize(1920, 1080)

        self.video_worker: VideoWorker | None = None
        self.video_thread: QThread | None = None
        self.imu_worker: ImuWorker | None = None
        self.imu_thread: QThread | None = None
        self._shutdown_requested = False
        self._stopping = False
        self._active_error = False

        self.preview = ImageView()
        self.pose_view = Pose3DWidget()
        self.device_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh Devices")
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.interval_spin = QDoubleSpinBox()
        self.accel_blend_spin = QDoubleSpinBox()
        self.gyro_only_check = QCheckBox("Gyro Only")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(300)

        self._build_ui()
        self._connect_signals()
        self._apply_args()
        self.refresh_devices()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)

        root_layout.addWidget(self._build_control_group(), stretch=0)

        viewer_layout = QHBoxLayout()
        viewer_layout.addWidget(self.preview, stretch=3)
        viewer_layout.addWidget(self.pose_view, stretch=2)
        root_layout.addLayout(viewer_layout, stretch=1)

        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        status_layout.addWidget(self.log_view)
        root_layout.addWidget(status_group, stretch=0)

        self.setCentralWidget(root)

    def _build_control_group(self) -> QGroupBox:
        group = QGroupBox("Device")
        layout = QHBoxLayout(group)

        self.device_combo.setMinimumWidth(520)

        self.interval_spin.setRange(1.0, 1000.0)
        self.interval_spin.setSingleStep(1.0)
        self.interval_spin.setDecimals(1)
        self.interval_spin.setSuffix(" ms")

        self.accel_blend_spin.setRange(0.0, 1.0)
        self.accel_blend_spin.setSingleStep(0.01)
        self.accel_blend_spin.setDecimals(3)

        form = QFormLayout()
        form.addRow("Camera", self.device_combo)
        form.addRow("IMU Interval", self.interval_spin)
        form.addRow("Accel Blend", self.accel_blend_spin)

        buttons = QVBoxLayout()
        buttons.addWidget(self.refresh_button)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.gyro_only_check)
        buttons.addStretch(1)

        layout.addLayout(form, stretch=1)
        layout.addLayout(buttons, stretch=0)
        self.stop_button.setEnabled(False)
        return group

    def _connect_signals(self) -> None:
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.start_button.clicked.connect(self.start_preview)
        self.stop_button.clicked.connect(self.stop_preview)

    def _apply_args(self) -> None:
        self.interval_spin.setValue(float(self.args.interval_ms))
        self.accel_blend_spin.setValue(float(self.args.accel_blend))
        self.gyro_only_check.setChecked(bool(self.args.gyro_only))

    def _append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)

    def _set_running_state(self, running: bool) -> None:
        has_device = self.device_combo.count() > 0
        self.refresh_button.setEnabled(not running)
        self.start_button.setEnabled(not running and has_device)
        self.stop_button.setEnabled(running)
        self.device_combo.setEnabled(not running)
        self.interval_spin.setEnabled(not running)
        self.accel_blend_spin.setEnabled(not running)
        self.gyro_only_check.setEnabled(not running)

    def refresh_devices(self) -> None:
        try:
            from arducam_uvc_stereo_sdk import scan_devices

            entries = build_device_entries(scan_devices())
        except Exception as exc:
            self.device_combo.clear()
            self._set_running_state(False)
            self.preview.clear_frame("Device scan failed")
            self._append_log(f"Device refresh failed: {exc}")
            QMessageBox.critical(self, "Device Scan Failed", str(exc))
            return

        previous = self.device_combo.currentData()
        self.device_combo.clear()

        for entry in entries:
            if not entry["has_capture_source"]:
                continue
            self.device_combo.addItem(entry["label"], entry["scan_index"])

        if self.device_combo.count() == 0:
            self.preview.clear_frame("No usable stereo devices found")
            self.pose_view.clear_pose()
            self._append_log("No usable stereo devices found")
            self._set_running_state(False)
            return

        if previous is not None:
            index = self.device_combo.findData(previous)
            if index >= 0:
                self.device_combo.setCurrentIndex(index)

        self._append_log(f"Found {self.device_combo.count()} usable device(s)")
        self._set_running_state(False)

    def start_preview(self) -> None:
        if self.video_worker is not None or self.imu_worker is not None:
            return

        scan_index = self.device_combo.currentData()
        if scan_index is None:
            QMessageBox.warning(
                self, "No Device Selected", "Please select a usable device first"
            )
            return

        self._stopping = False
        self._active_error = False
        self.preview.clear_frame("Starting camera...")
        self.pose_view.clear_pose()
        self._append_log(f"Starting preview for device[{scan_index}]")
        self._set_running_state(True)

        self.video_thread = QThread(self)
        self.video_worker = VideoWorker(int(scan_index))
        self.video_worker.moveToThread(self.video_thread)
        self.video_thread.started.connect(self.video_worker.run)
        self.video_worker.frame_ready.connect(self.on_frame_ready)
        self.video_worker.status.connect(self.on_worker_status)
        self.video_worker.error.connect(self.on_worker_error)
        self.video_worker.finished.connect(self.on_video_finished)
        self.video_worker.finished.connect(self.video_worker.deleteLater)
        self.video_worker.finished.connect(self.video_thread.quit)
        self.video_thread.finished.connect(self.on_video_thread_finished)
        self.video_thread.finished.connect(self.video_thread.deleteLater)

        self.imu_thread = QThread(self)
        self.imu_worker = ImuWorker(
            int(scan_index),
            interval_ms=self.interval_spin.value(),
            accel_blend=self.accel_blend_spin.value(),
            use_accel=not self.gyro_only_check.isChecked(),
        )
        self.imu_worker.moveToThread(self.imu_thread)
        self.imu_thread.started.connect(self.imu_worker.run)
        self.imu_worker.pose_ready.connect(self.on_pose_ready)
        self.imu_worker.status.connect(self.on_worker_status)
        self.imu_worker.error.connect(self.on_worker_error)
        self.imu_worker.finished.connect(self.on_imu_finished)
        self.imu_worker.finished.connect(self.imu_worker.deleteLater)
        self.imu_worker.finished.connect(self.imu_thread.quit)
        self.imu_thread.finished.connect(self.on_imu_thread_finished)
        self.imu_thread.finished.connect(self.imu_thread.deleteLater)

        self.video_thread.start()
        self.imu_thread.start()

    def stop_preview(self) -> None:
        if self.video_worker is None and self.imu_worker is None:
            return

        self._stopping = True
        self.stop_button.setEnabled(False)
        self._append_log("Stopping preview...")
        if self.video_worker is not None:
            self.video_worker.stop()
        if self.imu_worker is not None:
            self.imu_worker.stop()

    @pyqtSlot(object)
    def on_frame_ready(self, frame_rgb: Any) -> None:
        self.preview.set_frame(frame_rgb)

    @pyqtSlot(object, object, float, float)
    def on_pose_ready(
        self, pose: Any, sample: Any, read_ms: float, interval_ms: float
    ) -> None:
        self.pose_view.set_pose(pose, sample, read_ms, interval_ms)

    @pyqtSlot(str)
    def on_worker_status(self, message: str) -> None:
        self._append_log(message)

    @pyqtSlot(str)
    def on_worker_error(self, message: str) -> None:
        self._append_log(message)
        if not self._active_error:
            self._active_error = True
            QMessageBox.critical(self, "Preview Error", message)
        self.stop_preview()

    @pyqtSlot()
    def on_video_finished(self) -> None:
        self.video_worker = None
        self._maybe_all_workers_finished()

    @pyqtSlot()
    def on_imu_finished(self) -> None:
        self.imu_worker = None
        self._maybe_all_workers_finished()

    @pyqtSlot()
    def on_video_thread_finished(self) -> None:
        self.video_thread = None

    @pyqtSlot()
    def on_imu_thread_finished(self) -> None:
        self.imu_thread = None

    def _maybe_all_workers_finished(self) -> None:
        if self.video_worker is not None or self.imu_worker is not None:
            return
        if not self._shutdown_requested:
            self._set_running_state(False)
            if self._stopping and not self._active_error:
                self._append_log("Preview stopped")
        self._stopping = False

    def shutdown(self) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        self.stop_preview()
        for thread in (self.video_thread, self.imu_thread):
            if thread is not None:
                thread.quit()
                thread.wait(3000)

    def closeEvent(self, event: Any) -> None:
        self.shutdown()
        super().closeEvent(event)


def install_sigint_handler(window: StereoImuMainWindow) -> tuple[QTimer, Any]:
    def handle_sigint(_signum: int, _frame: Any) -> None:
        print("\n[INFO] Received Ctrl+C, shutting down...", file=sys.stderr)
        QTimer.singleShot(0, window.close)

    heartbeat = QTimer()
    heartbeat.setInterval(200)
    heartbeat.timeout.connect(lambda: None)
    heartbeat.start()

    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handle_sigint)
    return heartbeat, previous_handler


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    app.setStyle("Fusion")

    window = StereoImuMainWindow(args)
    app.aboutToQuit.connect(window.shutdown)
    sigint_heartbeat, previous_sigint_handler = install_sigint_handler(window)
    app._sigint_heartbeat = sigint_heartbeat
    window.show()
    try:
        return app.exec()
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)


if __name__ == "__main__":
    raise SystemExit(main())
