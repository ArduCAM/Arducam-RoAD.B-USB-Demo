"""Python bootstrap for the C++ stereo camera node.

This module keeps the Python-only Arducam SDK responsibilities local to startup:
device enumeration, device selection, and reading calibration JSON from flash.
The high-rate capture, split, CameraInfo construction, and publishing path lives
in the C++ node.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from arducam_uvc_stereo_sdk import OpenCvBackend, UVCStereo

from .calibration import build_camera_info_pair, parse_stereo_calibration
from .device_selector import SelectionError, SelectionParameters, describe_device, format_device_table, select_device
from .naming import auto_namespace_from_ros_args, camera_name_from_namespace

PACKAGE_NAME = "arducam_uvc_stereo_ros"
CPP_EXECUTABLE = "stereo_camera_cpp_node"


@dataclass(frozen=True)
class PreparedCppNode:
    """Prepared parameter set for launching the C++ publisher node."""

    parameters: dict
    namespace: str
    selected_device: object
    selection_mode: str
    warnings: tuple[str, ...]


def prepare_cpp_node(args: Optional[Sequence[str]] = None) -> PreparedCppNode:
    """Resolve SDK-only inputs and build the C++ node parameter handoff."""

    cli_args = sys.argv[1:] if args is None else list(args)
    cli_params = _parse_ros_params(cli_args)
    namespace = _namespace_from_args(cli_args)
    camera_name = camera_name_from_namespace(str(cli_params.get("camera_name", "")), namespace)

    sdk = UVCStereo()
    devices = sdk.scan()
    if not devices:
        raise RuntimeError("No Arducam UVC stereo devices were found.")

    selection_params = SelectionParameters(
        serial_number=str(cli_params.get("serial_number", "")),
        video_node=str(cli_params.get("video_node", "")),
        bus_number=_param_int(cli_params, "bus_number", -1),
        device_address=_param_int(cli_params, "device_address", -1),
        vid=_param_int(cli_params, "vid", -1),
        pid=_param_int(cli_params, "pid", -1),
        device_index=_param_int(cli_params, "device_index", -1),
    )

    try:
        selection = select_device(devices, selection_params)
    except SelectionError as exc:
        raise RuntimeError(f"{exc}\nDetected devices:\n{format_device_table(devices)}") from exc

    version, json_text = sdk.read_json(device=selection.device)
    capture_params = _capture_source_parameters(selection.device)

    translation_scale_to_meter = _required_param_float(cli_params, "translation_scale_to_meter") if "translation_scale_to_meter" in cli_params else 0.01
    calibration = parse_stereo_calibration(
        json_text=json_text,
        version=int(version),
        translation_scale_to_meter=translation_scale_to_meter,
    )

    combined_width = _required_param_int(cli_params, "combined_width") if "combined_width" in cli_params else 2 * calibration.left.width
    combined_height = _required_param_int(cli_params, "combined_height") if "combined_height" in cli_params else calibration.left.height
    if combined_width <= 0:
        combined_width = 2 * calibration.left.width
    if combined_height <= 0:
        combined_height = calibration.left.height
    if combined_width % 2 != 0:
        raise ValueError(f"combined_width must be even, got {combined_width}.")

    left_frame_id = str(cli_params.get("left_frame_id", "")) or f"{camera_name}_left_optical_frame"
    right_frame_id = str(cli_params.get("right_frame_id", "")) or f"{camera_name}_right_optical_frame"
    camera_info_pair = build_camera_info_pair(
        calibration=calibration,
        actual_width=combined_width // 2,
        actual_height=combined_height,
        left_frame_id=left_frame_id,
        right_frame_id=right_frame_id,
    )

    parameters = {
        "camera_name": camera_name,
        "left_frame_id": left_frame_id,
        "right_frame_id": right_frame_id,
        "calibration_version": int(version),
        "capture_source_description": capture_params["capture_source_description"],
        "combined_width": combined_width,
        "combined_height": combined_height,
        "translation_scale_to_meter": translation_scale_to_meter,
        **_camera_info_parameters(camera_info_pair),
        **capture_params,
    }

    float_passthrough_names = ("fps", "post_calibration_open_delay_sec", "retry_open_interval_sec")
    for name in float_passthrough_names:
        if name in cli_params:
            parameters[name] = _required_param_float(cli_params, name)

    string_passthrough_names = ("pixel_format", "qos_reliability")
    for name in string_passthrough_names:
        if name in cli_params:
            parameters[name] = cli_params[name]

    return PreparedCppNode(
        parameters=parameters,
        namespace=namespace,
        selected_device=selection.device,
        selection_mode=selection.mode,
        warnings=tuple(selection.warnings),
    )


def build_cpp_ros_args(prepared: PreparedCppNode, original_args: Optional[Sequence[str]] = None) -> list[str]:
    """Build command-line ROS arguments for the prepared C++ node."""

    original = sys.argv[1:] if original_args is None else list(original_args)
    forwarded = _strip_selection_only_params(original)

    has_ros_args = "--ros-args" in forwarded
    result = list(forwarded)
    if not has_ros_args:
        result.append("--ros-args")

    if prepared.namespace and not _has_namespace_remap(result):
        result.extend(["-r", f"__ns:=/{prepared.namespace.strip('/')}"])

    for name, value in prepared.parameters.items():
        result.extend(["-p", f"{name}:={_format_param_value(value)}"])

    return result


def main(args: Optional[Sequence[str]] = None) -> int:
    """Prepare SDK data, then replace this process with the C++ node."""

    cli_args = sys.argv[1:] if args is None else list(args)
    try:
        prepared = prepare_cpp_node(cli_args)
    except Exception as exc:  # pragma: no cover - startup error path
        print(f"Failed to prepare C++ stereo camera node: {exc}", file=sys.stderr)
        return 1

    print(
        f"Selected device via {prepared.selection_mode}: {describe_device(prepared.selected_device)}",
        file=sys.stderr,
        flush=True,
    )
    for warning in prepared.warnings:
        print(f"Warning: {warning}", file=sys.stderr, flush=True)

    cpp_args = build_cpp_ros_args(prepared, cli_args)
    command = ["ros2", "run", PACKAGE_NAME, CPP_EXECUTABLE, *cpp_args]
    os.execvp(command[0], command)
    return 1



def _camera_info_parameters(pair) -> dict:
    return {
        "left_camera_info_width": int(pair.left.width),
        "left_camera_info_height": int(pair.left.height),
        "left_camera_info_distortion_model": str(pair.left.distortion_model),
        "left_camera_info_d": [float(value) for value in pair.left.d],
        "left_camera_info_k": [float(value) for value in pair.left.k],
        "left_camera_info_r": [float(value) for value in pair.left.r],
        "left_camera_info_p": [float(value) for value in pair.left.p],
        "right_camera_info_width": int(pair.right.width),
        "right_camera_info_height": int(pair.right.height),
        "right_camera_info_distortion_model": str(pair.right.distortion_model),
        "right_camera_info_d": [float(value) for value in pair.right.d],
        "right_camera_info_k": [float(value) for value in pair.right.k],
        "right_camera_info_r": [float(value) for value in pair.right.r],
        "right_camera_info_p": [float(value) for value in pair.right.p],
    }

def _namespace_from_args(args: Sequence[str]) -> str:
    explicit = _namespace_remap_value(args)
    if explicit:
        return explicit.strip().strip("/")
    return auto_namespace_from_ros_args(args).strip().strip("/")


def _write_calibration_json(json_text: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="arducam_uvc_stereo_calib_",
        suffix=".json",
        delete=False,
    )
    with handle:
        handle.write(json_text)
    return Path(handle.name)


def _capture_source_parameters(device: object) -> dict:
    video_node = str(getattr(device, "video_node", ""))
    backend_indices = getattr(device, "opencv_backend_indices", None)

    # Prefer the stable /dev/videoX path for the C++ node. Some OpenCV/V4L2
    # builds on Jetson can segfault when opening by backend-specific integer index.
    if video_node:
        return {
            "use_capture_index": False,
            "capture_index": -1,
            "video_node": video_node,
            "capture_source_description": f"video node {video_node}",
        }

    if isinstance(backend_indices, dict):
        index = backend_indices.get(OpenCvBackend.CAP_V4L2)
        if index is None:
            for backend, backend_index in backend_indices.items():
                try:
                    if int(backend) == int(OpenCvBackend.CAP_V4L2):
                        index = backend_index
                        break
                except (TypeError, ValueError):
                    continue
        if index is not None:
            index = int(index)
            return {
                "use_capture_index": True,
                "capture_index": index,
                "video_node": video_node,
                "capture_source_description": f"OpenCV CAP_V4L2 index {index}",
            }

    return {
        "use_capture_index": False,
        "capture_index": -1,
        "video_node": video_node,
        "capture_source_description": f"video node {video_node}",
    }


def _parse_ros_params(args: Sequence[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for assignment in _option_values(args, ("-p", "--param")):
        if ":=" not in assignment:
            continue
        name, value = assignment.split(":=", 1)
        name = _base_assignment_name(name)
        params[name] = value.strip().strip("\"'")
    return params


def _strip_selection_only_params(args: Sequence[str]) -> list[str]:
    selection_only = {
        "serial_number",
        "bus_number",
        "device_address",
        "vid",
        "pid",
        "device_index",
    }
    result: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in ("-p", "--param") and index + 1 < len(args):
            assignment = args[index + 1]
            if _param_assignment_name(assignment) in selection_only:
                index += 2
                continue
            result.extend([token, assignment])
            index += 2
            continue
        if token.startswith("--param="):
            assignment = token[len("--param=") :]
            if _param_assignment_name(assignment) in selection_only:
                index += 1
                continue
        if token.startswith("-p") and len(token) > 2:
            assignment = token[2:]
            if _param_assignment_name(assignment) in selection_only:
                index += 1
                continue
        result.append(token)
        index += 1
    return result


def _param_assignment_name(assignment: str) -> str:
    if ":=" not in assignment:
        return ""
    name, _ = assignment.split(":=", 1)
    return _base_assignment_name(name)


def _base_assignment_name(name: str) -> str:
    name = name.strip()
    if "." in name:
        name = name.rsplit(".", 1)[1]
    if ":" in name:
        name = name.rsplit(":", 1)[1]
    return name


def _namespace_remap_value(args: Sequence[str]) -> str:
    for rule in _option_values(args, ("-r", "--remap")):
        if ":=" not in rule:
            continue
        source, target = rule.split(":=", 1)
        source = source.strip()
        if source == "__ns" or source.endswith(":__ns"):
            return target
    return ""


def _has_namespace_remap(args: Sequence[str]) -> bool:
    return bool(_namespace_remap_value(args))


def _option_values(args: Sequence[str], options: Iterable[str]) -> Iterable[str]:
    option_set = tuple(options)
    index = 0
    while index < len(args):
        token = args[index]
        if token in option_set:
            if index + 1 < len(args):
                yield args[index + 1]
            index += 2
            continue
        for option in option_set:
            if token.startswith(f"{option}="):
                yield token[len(option) + 1 :]
                break
            if len(option) == 2 and token.startswith(option) and len(token) > 2:
                yield token[len(option) :]
                break
        index += 1


def _param_int(params: dict[str, str], name: str, default: int) -> int:
    try:
        return int(str(params.get(name, default)), 0)
    except (TypeError, ValueError):
        return default



def _required_param_int(params: dict[str, str], name: str) -> int:
    try:
        return int(str(params[name]), 0)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Parameter {name} must be an integer, got {params.get(name)!r}.") from exc


def _required_param_float(params: dict[str, str], name: str) -> float:
    try:
        return float(str(params[name]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Parameter {name} must be a float, got {params.get(name)!r}.") from exc

def _format_param_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_param_value(item) for item in value) + "]"
    return str(value)
