"""ROS 2 node for publishing Arducam UVC stereo cameras."""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional, Sequence

import rclpy
from arducam_uvc_stereo_sdk import OpenCvBackend, UVCStereo
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from .calibration import CalibrationError, CameraInfoPair, StereoCalibration, build_camera_info_pair, clone_camera_info, parse_stereo_calibration
from .capture_backend import CaptureError, FrameFormatError, VideoCaptureBackend, describe_capture_source, split_stereo_frame
from .device_selector import SelectionError, SelectionParameters, describe_device, format_device_table, select_device
from .naming import auto_namespace_from_ros_args, camera_name_from_namespace


class StereoCameraNode(Node):
    """Publish synchronized stereo images and camera_info messages."""

    def __init__(
        self,
        *,
        node_name: str = "stereo_camera_node",
        namespace: str = "",
        parameter_overrides: Optional[Sequence] = None,
    ) -> None:
        super().__init__(node_name, namespace=namespace, parameter_overrides=parameter_overrides or [])
        default_camera_name = camera_name_from_namespace("", self.get_namespace())

        self.declare_parameters(
            namespace="",
            parameters=[
                ("camera_name", default_camera_name),
                ("serial_number", ""),
                ("video_node", ""),
                ("bus_number", -1),
                ("device_address", -1),
                ("vid", -1),
                ("pid", -1),
                ("device_index", -1),
                ("combined_width", 0),
                ("combined_height", 0),
                ("fps", 30.0),
                ("pixel_format", "MJPG"),
                ("qos_reliability", "best_effort"),
                ("translation_scale_to_meter", 0.01),
                ("post_calibration_open_delay_sec", 3.0),
                ("retry_open_interval_sec", 1.0),
                ("left_frame_id", ""),
                ("right_frame_id", ""),
            ],
        )

        self.camera_name = str(self.get_parameter("camera_name").value)
        self.left_frame_id = str(self.get_parameter("left_frame_id").value) or f"{self.camera_name}_left_optical_frame"
        self.right_frame_id = str(self.get_parameter("right_frame_id").value) or f"{self.camera_name}_right_optical_frame"
        self.retry_open_interval_sec = float(self.get_parameter("retry_open_interval_sec").value)
        self.pixel_format = str(self.get_parameter("pixel_format").value).upper()
        self.qos_reliability = str(self.get_parameter("qos_reliability").value).lower()
        self.fps = float(self.get_parameter("fps").value)
        self._requested_combined_width = int(self.get_parameter("combined_width").value)
        self._requested_combined_height = int(self.get_parameter("combined_height").value)
        self._translation_scale_to_meter = float(self.get_parameter("translation_scale_to_meter").value)
        self._post_calibration_open_delay_sec = float(self.get_parameter("post_calibration_open_delay_sec").value)
        self._publisher_qos = build_publisher_qos(self.qos_reliability)

        self.bridge = CvBridge()
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._fatal_error_message: Optional[str] = None
        self._fatal_error_reported = False
        self._backend: Optional[VideoCaptureBackend] = None
        self._calibration: Optional[StereoCalibration] = None
        self._camera_info_pair: Optional[CameraInfoPair] = None
        self._publish_size = None
        self._selected_device = None
        self._calibration_version = None
        self._capture_source = None
        self._capture_source_description = ""

        self.left_image_pub = self.create_publisher(Image, "left/image_raw", self._publisher_qos)
        self.right_image_pub = self.create_publisher(Image, "right/image_raw", self._publisher_qos)
        self.left_info_pub = self.create_publisher(CameraInfo, "left/camera_info", self._publisher_qos)
        self.right_info_pub = self.create_publisher(CameraInfo, "right/camera_info", self._publisher_qos)
        self._health_timer = self.create_timer(0.2, self._check_background_health)

        self._sdk = UVCStereo()
        self._initialize_pipeline()

        self._capture_thread = threading.Thread(target=self._capture_loop, name="stereo_capture_thread", daemon=True)
        self._capture_thread.start()

    @property
    def fatal_error_message(self) -> Optional[str]:
        return self._fatal_error_message

    def destroy_node(self) -> bool:
        self._stop_event.set()
        if self._capture_thread is not None and self._capture_thread.is_alive() and threading.current_thread() is not self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._backend is not None:
            self._backend.release()
            self._backend = None
        return super().destroy_node()

    def _initialize_pipeline(self) -> None:
        devices = self._sdk.scan()
        if not devices:
            raise RuntimeError("No Arducam UVC stereo devices were found.")

        selection = self._select_device(devices)
        self._selected_device = selection.device

        for warning in selection.warnings:
            self.get_logger().warning(warning)

        self.get_logger().info(
            f"Selected device via {selection.mode}: {describe_device(self._selected_device)}"
        )
        self.get_logger().info(f"Publisher QoS reliability: {self.qos_reliability}")
        self._capture_source, self._capture_source_description = self._resolve_capture_source(self._selected_device)
        self.get_logger().info(f"Capture source: {self._capture_source_description}")

        version, json_text = self._sdk.read_json(device=self._selected_device)
        self._calibration_version = int(version)
        self._calibration = parse_stereo_calibration(
            json_text=json_text,
            version=self._calibration_version,
            translation_scale_to_meter=self._translation_scale_to_meter,
        )

        if self._post_calibration_open_delay_sec > 0.0:
            self.get_logger().info(
                f"Waiting {self._post_calibration_open_delay_sec:.1f}s after calibration read before opening UVC stream."
            )
            time.sleep(self._post_calibration_open_delay_sec)

        self._open_capture(initial=True)

        assert self._calibration is not None
        assert self._publish_size is not None
        self.get_logger().info(
            "Calibration version=%d raw_size=%dx%d baseline_m=%.6f publish_size=%dx%d capture_source=%s"
            % (
                self._calibration.version,
                self._calibration.left.width,
                self._calibration.left.height,
                self._calibration.baseline_m,
                self._publish_size[0],
                self._publish_size[1],
                self._capture_source_description,
            )
        )

    def _select_device(self, devices):
        params = SelectionParameters(
            serial_number=str(self.get_parameter("serial_number").value),
            video_node=str(self.get_parameter("video_node").value),
            bus_number=int(self.get_parameter("bus_number").value),
            device_address=int(self.get_parameter("device_address").value),
            vid=int(self.get_parameter("vid").value),
            pid=int(self.get_parameter("pid").value),
            device_index=int(self.get_parameter("device_index").value),
        )

        try:
            return select_device(devices, params)
        except SelectionError as exc:
            raise RuntimeError(f"{exc}\nDetected devices:\n{format_device_table(devices)}") from exc

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._backend is None:
                    self._reopen_capture_loop()
                    continue

                ok, frame = self._backend.read()
                if not ok or frame is None:
                    self.get_logger().warning("Frame read failed, attempting to reopen capture.")
                    self._drop_capture()
                    continue

                left_frame, right_frame, encoding = split_stereo_frame(frame)
                self._publish_frame_pair(left_frame, right_frame, encoding)
            except FrameFormatError as exc:
                self._set_fatal_error(f"Invalid stereo frame format: {exc}")
            except CalibrationError as exc:
                self._set_fatal_error(f"Calibration became incompatible with capture output: {exc}")
            except CaptureError as exc:
                self.get_logger().warning(f"Capture error: {exc}; reopening stream.")
                self._drop_capture()
            except Exception as exc:  # pragma: no cover - unexpected runtime guard
                self._set_fatal_error(f"Unhandled capture loop error: {exc}")

    def _reopen_capture_loop(self) -> None:
        while not self._stop_event.is_set() and self._backend is None:
            try:
                self._open_capture(initial=False)
                self.get_logger().info("Reopened capture stream.")
                return
            except (CalibrationError, FrameFormatError) as exc:
                self._set_fatal_error(f"Reopened capture produced incompatible output: {exc}")
                return
            except CaptureError as exc:
                self.get_logger().warning(
                    f"Failed to reopen {self._capture_source_description}: {exc}. "
                    f"Retrying in {self.retry_open_interval_sec:.1f}s."
                )
                time.sleep(self.retry_open_interval_sec)

    def _open_capture(self, initial: bool) -> None:
        if self._calibration is None:
            raise RuntimeError("Calibration must be loaded before opening capture.")

        combined_width = self._requested_combined_width or (2 * self._calibration.left.width)
        combined_height = self._requested_combined_height or self._calibration.left.height

        backend = VideoCaptureBackend(
            source=self._capture_source,
            combined_width=combined_width,
            combined_height=combined_height,
            fps=self.fps,
            pixel_format=self.pixel_format,
            source_description=self._capture_source_description,
        )
        try:
            settings = backend.open()
            frame = self._read_initial_frame(backend)
            left_frame, right_frame, _ = split_stereo_frame(frame)
            pair = build_camera_info_pair(
                calibration=self._calibration,
                actual_width=int(left_frame.shape[1]),
                actual_height=int(left_frame.shape[0]),
                left_frame_id=self.left_frame_id,
                right_frame_id=self.right_frame_id,
            )
        except Exception:
            backend.release()
            raise

        previous_size = self._publish_size
        self._backend = backend
        self._camera_info_pair = pair
        self._publish_size = (pair.width, pair.height)

        if initial:
            self.get_logger().info(
                "Opened %s requested=%dx%d@%.2f actual=%dx%d@%.2f fourcc=%s"
                % (
                    self._capture_source_description,
                    combined_width,
                    combined_height,
                    self.fps,
                    settings.width,
                    settings.height,
                    settings.fps,
                    settings.pixel_format,
                )
            )
            return

        if previous_size != self._publish_size:
            self.get_logger().warning(
                f"Capture publish size changed from {previous_size} to {self._publish_size} after reopen."
            )

    def _drop_capture(self) -> None:
        if self._backend is not None:
            self._backend.release()
            self._backend = None
        time.sleep(self.retry_open_interval_sec)

    def _read_initial_frame(self, backend: VideoCaptureBackend):
        for _ in range(20):
            ok, frame = backend.read()
            if ok and frame is not None:
                return frame
            time.sleep(0.05)
        backend.release()
        raise CaptureError(f"Timed out waiting for a valid frame from {self._capture_source_description}.")

    def _publish_frame_pair(self, left_frame, right_frame, encoding: str) -> None:
        assert self._camera_info_pair is not None

        stamp = self.get_clock().now().to_msg()

        left_image = self.bridge.cv2_to_imgmsg(left_frame, encoding=encoding)
        left_image.header.stamp = stamp
        left_image.header.frame_id = self.left_frame_id

        right_image = self.bridge.cv2_to_imgmsg(right_frame, encoding=encoding)
        right_image.header.stamp = stamp
        right_image.header.frame_id = self.right_frame_id

        left_info = clone_camera_info(self._camera_info_pair.left, stamp)
        right_info = clone_camera_info(self._camera_info_pair.right, stamp)

        self.left_image_pub.publish(left_image)
        self.right_image_pub.publish(right_image)
        self.left_info_pub.publish(left_info)
        self.right_info_pub.publish(right_info)

    def _set_fatal_error(self, message: str) -> None:
        self._fatal_error_message = message
        self._stop_event.set()
        if self._backend is not None:
            self._backend.release()
            self._backend = None

    def _check_background_health(self) -> None:
        if self._fatal_error_message is None or self._fatal_error_reported:
            return
        self._fatal_error_reported = True
        self.get_logger().error(self._fatal_error_message)
        if rclpy.ok():
            rclpy.shutdown()

    def _resolve_capture_source(self, device) -> tuple[object, str]:
        try:
            backend_indices = getattr(device, "opencv_backend_indices", None)
        except Exception:
            backend_indices = None

        if isinstance(backend_indices, dict):
            index = backend_indices.get(OpenCvBackend.CAP_V4L2)
            if index is None:
                for backend, backend_index in backend_indices.items():
                    try:
                        if int(backend) == int(OpenCvBackend.CAP_V4L2):
                            index = int(backend_index)
                            break
                    except (TypeError, ValueError):
                        continue
            if index is not None:
                index = int(index)
                return index, f"OpenCV CAP_V4L2 index {index} (device {device.video_node})"

        return device.video_node, f"video node {describe_capture_source(device.video_node)}"


def build_publisher_qos(reliability_mode: str) -> QoSProfile:
    """Build publisher QoS from a reliability mode string."""

    mode = reliability_mode.lower()
    if mode == "best_effort":
        reliability = ReliabilityPolicy.BEST_EFFORT
    elif mode == "reliable":
        reliability = ReliabilityPolicy.RELIABLE
    else:
        raise ValueError(
            f"Unsupported qos_reliability {reliability_mode!r}. Expected 'best_effort' or 'reliable'."
        )

    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=qos_profile_sensor_data.depth,
        reliability=reliability,
        durability=DurabilityPolicy.VOLATILE,
    )


def main(args=None) -> int:
    cli_args = sys.argv[1:] if args is None else list(args)
    namespace = auto_namespace_from_ros_args(cli_args)

    rclpy.init(args=args)
    node = None
    exit_code = 0

    try:
        node = StereoCameraNode(namespace=namespace)
        rclpy.spin(node)
        if node.fatal_error_message:
            exit_code = 1
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        exit_code = 1
        if node is not None:
            node.get_logger().error(str(exc))
        else:
            print(f"Failed to start stereo camera node: {exc}", file=sys.stderr)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
