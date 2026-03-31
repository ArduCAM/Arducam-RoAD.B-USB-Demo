#!/usr/bin/env python3
import argparse
import json
import sys

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arducam UVC Stereo live rectification demo")
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Index in UVCStereo.scan() result. Default: prompt in CLI when multiple devices are detected",
    )
    parser.add_argument(
        "--video-node",
        type=str,
        default=None,
        help="Optional video source override, e.g. /dev/video0",
    )
    return parser.parse_args()


def format_device(dev):
    parts = []
    vid = getattr(dev, "vid", None)
    pid = getattr(dev, "pid", None)
    video_node = getattr(dev, "video_node", None)
    bus_number = getattr(dev, "bus_number", None)
    device_address = getattr(dev, "device_address", None)

    if vid is not None:
        parts.append(f"vid=0x{vid:04x}")
    if pid is not None:
        parts.append(f"pid=0x{pid:04x}")
    if video_node:
        parts.append(f"node={video_node}")
    if bus_number is not None:
        parts.append(f"bus={bus_number}")
    if device_address is not None:
        parts.append(f"address={device_address}")

    return " ".join(parts) if parts else str(dev)


def get_opencv_candidates(dev):
    candidates = []
    seen = set()
    raw = getattr(dev, "opencv", None)

    if isinstance(raw, dict):
        raw = [raw]

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            for backend_name, source_id in item.items():
                backend_name = str(backend_name)
                try:
                    source_id = int(source_id)
                except (TypeError, ValueError):
                    continue
                key = (backend_name, source_id)
                if key in seen:
                    continue
                if getattr(cv2, backend_name, None) is None:
                    continue
                seen.add(key)
                candidates.append((backend_name, source_id))
        if candidates:
            return candidates

    backend_indices = getattr(dev, "opencv_backend_indices", None)
    if isinstance(backend_indices, dict):
        for backend, source_id in backend_indices.items():
            backend_name = getattr(backend, "name", str(backend))
            try:
                source_id = int(source_id)
            except (TypeError, ValueError):
                continue
            key = (backend_name, source_id)
            if key in seen:
                continue
            if getattr(cv2, backend_name, None) is None:
                continue
            seen.add(key)
            candidates.append((backend_name, source_id))

    return candidates


def get_capture_candidates(dev):
    candidates = []
    seen = set()

    for backend_name, source_id in get_opencv_candidates(dev):
        backend_id = getattr(cv2, backend_name, None)
        if backend_id is None:
            continue
        key = ("opencv", backend_name, source_id)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "kind": "opencv",
            "backend_name": backend_name,
            "backend_id": int(backend_id),
            "source": source_id,
            "label": f"{backend_name}:{source_id}",
        })

    video_node = getattr(dev, "video_node", None)
    if video_node:
        backend_id = getattr(cv2, "CAP_V4L2", None)
        key = ("video_node", video_node)
        if key not in seen:
            seen.add(key)
            candidates.append({
                "kind": "video_node",
                "backend_name": "CAP_V4L2" if backend_id is not None else None,
                "backend_id": int(backend_id) if backend_id is not None else None,
                "source": video_node,
                "label": f"video_node:{video_node}",
            })

    return candidates


def build_device_entries(devices):
    entries = []
    for scan_index, dev in enumerate(devices):
        capture_candidates = get_capture_candidates(dev)
        entries.append({
            "scan_index": scan_index,
            "dev": dev,
            "capture_desc": "none" if not capture_candidates
            else ", ".join(candidate["label"] for candidate in capture_candidates),
            "has_capture_source": bool(capture_candidates),
        })
    return entries


def choose_device_via_cli(entries):
    if not sys.stdin.isatty():
        raise RuntimeError(
            "multiple devices detected in non-interactive mode, use --device-index"
        )

    valid_entries = {str(entry["scan_index"]): entry for entry in entries}
    valid_indices = ", ".join(valid_entries.keys())
    print("Multiple devices detected. Type the device index to select it.")
    print(f"Available indices: {valid_indices}")

    while True:
        try:
            choice = input("device index> ").strip()
        except EOFError as exc:
            raise RuntimeError("device selection cancelled") from exc

        if not choice:
            print(f"[WARN] Please enter one of: {valid_indices}")
            continue
        if choice.lower() in {"q", "quit", "exit"}:
            raise RuntimeError("device selection cancelled")
        if choice not in valid_entries:
            print(f"[WARN] Invalid device index '{choice}', valid choices: {valid_indices}")
            continue
        return valid_entries[choice]


def select_device(sdk, device_index=None, require_capture_source=False):
    devices = sdk.scan()
    if not devices:
        raise RuntimeError("no devices found")

    entries = build_device_entries(devices)
    candidates = []
    for entry in entries:
        note = ""
        if require_capture_source and not entry["has_capture_source"]:
            note = " [skip: no usable capture source]"
        print(
            f"device[{entry['scan_index']}]: {format_device(entry['dev'])} "
            f"| capture={entry['capture_desc']}{note}"
        )
        if not require_capture_source or entry["has_capture_source"]:
            candidates.append(entry)

    if device_index is not None:
        if device_index < 0 or device_index >= len(entries):
            raise RuntimeError(
                f"device index {device_index} is out of range for {len(entries)} detected device(s)"
            )
        selected = entries[device_index]
        if require_capture_source and not selected["has_capture_source"]:
            raise RuntimeError(
                f"device index {device_index} does not expose a usable capture source"
            )
        print(f"selected device: {format_device(selected['dev'])}")
        return selected["dev"]

    if candidates:
        if len(candidates) == 1:
            selected = candidates[0]
        else:
            selected = choose_device_via_cli(candidates)
        print(f"selected device: {format_device(selected['dev'])}")
        return selected["dev"]

    raise RuntimeError("no detected device exposes a usable capture source")


def read_device_calibration(sdk, dev):
    try:
        version, json_text = sdk.read_json(device=dev)
    except Exception as exc:
        raise RuntimeError(f"failed to read calibration JSON from device: {exc}") from exc

    print(f"calibration version={version}")

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid calibration JSON: {exc}") from exc


def require_camera(camera_data, name):
    for camera in camera_data:
        if isinstance(camera, dict) and camera.get("name") == name:
            return camera
    raise RuntimeError(f"cameraData is missing '{name}' camera")


def to_matrix(data, shape, field_name):
    matrix = np.asarray(data, dtype=np.float64)
    if matrix.shape != shape:
        raise RuntimeError(f"{field_name} must have shape {shape}, got {matrix.shape}")
    return matrix


def to_dist_coeffs(data, field_name):
    coeffs = np.asarray(data, dtype=np.float64).reshape(-1, 1)
    if coeffs.size == 0:
        raise RuntimeError(f"{field_name} is empty")
    return coeffs


def extract_stereo_params(calibration):
    camera_data = calibration.get("cameraData")
    if not isinstance(camera_data, list):
        raise RuntimeError("calibration JSON is missing cameraData list")

    left = require_camera(camera_data, "left")
    right = require_camera(camera_data, "right")

    try:
        width_l = int(left["width"])
        height_l = int(left["height"])
        width_r = int(right["width"])
        height_r = int(right["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid width/height in calibration JSON: {exc}") from exc

    if width_l <= 0 or height_l <= 0 or width_r <= 0 or height_r <= 0:
        raise RuntimeError("width and height in calibration JSON must be positive")
    if width_l != width_r or height_l != height_r:
        raise RuntimeError(
            f"left/right image size mismatch in calibration JSON: {width_l}x{height_l} vs {width_r}x{height_r}"
        )

    try:
        K_l = to_matrix(left["intrinsicMatrix"], (3, 3), "left intrinsicMatrix")
        D_l = to_dist_coeffs(left["dist_coeff"], "left dist_coeff")
        K_r = to_matrix(right["intrinsicMatrix"], (3, 3), "right intrinsicMatrix")
        D_r = to_dist_coeffs(right["dist_coeff"], "right dist_coeff")
    except KeyError as exc:
        raise RuntimeError(f"missing calibration field: {exc}") from exc

    extrinsics = left.get("extrinsics")
    if not isinstance(extrinsics, dict):
        raise RuntimeError("left extrinsics is missing")

    try:
        R = to_matrix(extrinsics["rotationMatrix"], (3, 3), "left extrinsics.rotationMatrix")
        T = np.asarray(extrinsics["translation"], dtype=np.float64).reshape(3, 1)
    except KeyError as exc:
        raise RuntimeError(f"missing extrinsics field: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(f"invalid left extrinsics.translation: {exc}") from exc

    if T.shape != (3, 1):
        raise RuntimeError(f"left extrinsics.translation must have shape (3, 1), got {T.shape}")

    return {
        "img_size": (width_l, height_l),
        "K_l": K_l,
        "D_l": D_l,
        "K_r": K_r,
        "D_r": D_r,
        "R": R,
        "T": T,
    }


def compute_maps(params):
    img_size = params["img_size"]
    R1, R2, P1, P2, _Q, _roi1, _roi2 = cv2.stereoRectify(
        params["K_l"],
        params["D_l"],
        params["K_r"],
        params["D_r"],
        img_size,
        params["R"],
        params["T"],
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    map_l1, map_l2 = cv2.initUndistortRectifyMap(
        params["K_l"], params["D_l"], R1, P1, img_size, cv2.CV_32FC1
    )
    map_r1, map_r2 = cv2.initUndistortRectifyMap(
        params["K_r"], params["D_r"], R2, P2, img_size, cv2.CV_32FC1
    )

    return map_l1, map_l2, map_r1, map_r2


def open_camera(candidates, width, height):
    total_width = width * 2
    if not candidates:
        raise RuntimeError("no usable capture source")

    errors = []
    for candidate in candidates:
        label = candidate["label"]
        print(f"trying camera source: {label}")

        if candidate["backend_id"] is None:
            cap = cv2.VideoCapture(candidate["source"])
        else:
            cap = cv2.VideoCapture(candidate["source"], candidate["backend_id"])

        if not cap.isOpened():
            cap.release()
            errors.append(f"{label}: open failed")
            print(f"cannot open camera via {label}")
            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, total_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"camera opened via {label}: {actual_w}x{actual_h} MJPG")

        if actual_w != total_width or actual_h != height:
            cap.release()
            errors.append(
                f"{label}: requested {total_width}x{height}, got {actual_w}x{actual_h}"
            )
            print(f"size mismatch via {label}: requested {total_width}x{height}, got {actual_w}x{actual_h}")
            continue

        return cap

    raise RuntimeError(
        "cannot open camera from candidates: " + ", ".join(errors)
    )


def split_stereo_frame(frame, img_size):
    if frame is None or frame.size == 0:
        raise ValueError("empty frame")

    frame_h, frame_w = frame.shape[:2]
    if frame_w % 2 != 0:
        raise ValueError(f"invalid frame width: {frame_w}")

    half_w = frame_w // 2
    left = frame[:, :half_w]
    right = frame[:, frame_w - half_w :]

    expected_w, expected_h = img_size
    if frame_h != expected_h or half_w != expected_w:
        raise ValueError(
            "frame size mismatch with calibration JSON: "
            f"expected total {expected_w * 2}x{expected_h}, got {frame_w}x{frame_h}"
        )

    return left, right


def preview_loop(cap, maps, img_size):
    map_l1, map_l2, map_r1, map_r2 = maps
    window_name = "Live Rectified Stereo - [q]/[Esc] quit"

    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame from camera")

        left, right = split_stereo_frame(frame, img_size)
        left_rect = cv2.remap(left, map_l1, map_l2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, map_r1, map_r2, cv2.INTER_LINEAR)

        combined = np.hstack([left_rect, right_rect])
        scale = min(1.0, 1920.0 / combined.shape[1])
        if scale < 1.0:
            combined = cv2.resize(combined, None, fx=scale, fy=scale)

        cv2.imshow(window_name, combined)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break


def main():
    args = parse_args()
    cap = None

    try:
        from arducam_uvc_stereo_sdk import UVCStereo

        sdk = UVCStereo()
        dev = select_device(
            sdk,
            device_index=args.device_index,
            require_capture_source=args.video_node is None,
        )
        calibration = read_device_calibration(sdk, dev)
        params = extract_stereo_params(calibration)
        maps = compute_maps(params)

        if args.video_node:
            candidates = [{
                "kind": "video_node",
                "backend_name": "CAP_V4L2" if getattr(cv2, "CAP_V4L2", None) is not None else None,
                "backend_id": int(cv2.CAP_V4L2) if getattr(cv2, "CAP_V4L2", None) is not None else None,
                "source": args.video_node,
                "label": f"video_node:{args.video_node}",
            }]
        else:
            candidates = get_capture_candidates(dev)

        if not candidates:
            raise RuntimeError("selected device does not expose a usable capture source, use --video-node")

        print("available capture sources: " + ", ".join(candidate["label"] for candidate in candidates))
        cap = open_camera(candidates, *params["img_size"])
        preview_loop(cap, maps, params["img_size"])
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
