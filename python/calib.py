#!/usr/bin/env python3
"""
Arducam UVC Stereo Camera Calibration Tool

Captures stereo image pairs from a side-by-side UVC camera (2560x800 MJPG),
performs Charuco-based stereo calibration, and outputs a calib_example.json
compatible with arducam_uvc_stereo_sdk.

Detection pipeline:
  detectMarkers -> refineDetectedMarkers -> interpolateCornersCharuco
with percentage-based marker threshold and camera orientation check.

Usage:
    python calibrate.py
    python calibrate.py -s 2.5 -ms 1.8 -nx 11 -ny 8
    python calibrate.py --device-index 1
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from arducam_uvc_stereo_sdk import UVCStereo


def get_project_root():
    return Path(__file__).resolve().parent.parent


def get_default_session_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return get_project_root() / "dataset" / timestamp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arducam UVC Stereo Camera Charuco Calibration")
    default_session_dir = get_default_session_dir()
    parser.add_argument("-W", "--width", type=int, default=2560,
                        help="Total frame width (side-by-side). Default: 2560")
    parser.add_argument("-H", "--height", type=int, default=800,
                        help="Frame height. Default: 800")
    parser.add_argument("-nx", "--board-cols", type=int, default=11,
                        help="Charuco board squares in X. Default: 11")
    parser.add_argument("-ny", "--board-rows", type=int, default=8,
                        help="Charuco board squares in Y. Default: 8")
    parser.add_argument("-s", "--square-size", type=float, default=2.5,
                        help="Square size in cm. Default: 2.5")
    parser.add_argument("-ms", "--marker-size", type=float, default=1.8,
                        help="Marker size in cm. Default: 1.8")
    parser.add_argument("-c", "--count", type=int, default=20,
                        help="Target number of image pairs. Default: 20")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output JSON path. Default: <project_root>/dataset/<timestamp>/calib_result.json")
    parser.add_argument("-m", "--mode", type=str, default="capture+process",
                        choices=["capture+process", "capture", "process"],
                        help="Mode: capture+process, capture only, or process only. Default: capture+process")
    parser.add_argument("--dataset", type=str, default=str(default_session_dir),
                        help="Directory to save captured images and calibration result. Default: <project_root>/dataset/<timestamp>")
    parser.add_argument("-mdmp", "--minDetectedMarkersPercent", type=float, default=0.4,
                        help="Min percentage of detected markers to accept a frame. Default: 0.4")
    parser.add_argument("--device-index", type=int, default=None,
                        help="Index in UVCStereo.scan() result. Default: prompt in CLI when multiple devices are detected")
    args = parser.parse_args()
    if args.output is None:
        args.output = str(Path(args.dataset) / "calib_result.json")
    return args


def format_device(dev):
    """Format device info for user-facing logs."""
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
    """Return usable OpenCV capture candidates as (backend_name, source_id)."""
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
    """Return usable camera open candidates from SDK device info."""
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


def format_capture_candidates(dev):
    """Format usable capture candidates for logs."""
    candidates = get_capture_candidates(dev)
    if not candidates:
        return "none"
    return ", ".join(candidate["label"] for candidate in candidates)


def build_device_entries(devices):
    """Build device metadata used by logs and selection flows."""
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
    """Choose a device by typing its scan index in the terminal."""
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


def select_device(sdk, require_capture_source=False, device_index=None):
    """Select an SDK device, prompting when multiple candidates are available."""
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


def open_camera(dev, width, height):
    """Open camera with MJPG format and specified resolution."""
    candidates = get_capture_candidates(dev)
    if not candidates:
        raise RuntimeError("selected device does not expose a usable capture source")

    for candidate in candidates:
        label = candidate["label"]
        print(f"[INFO] Trying camera source: {label}")

        if candidate["backend_id"] is None:
            cap = cv2.VideoCapture(candidate["source"])
        else:
            cap = cv2.VideoCapture(candidate["source"], candidate["backend_id"])

        if not cap.isOpened():
            cap.release()
            print(f"[WARN] Cannot open camera via {label}")
            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[OK] Camera opened via {label}: {actual_w}x{actual_h} MJPG")
        if actual_w != width or actual_h != height:
            print(f"[WARN] Requested {width}x{height}, got {actual_w}x{actual_h}")
        return cap

    raise RuntimeError(
        f"cannot open camera for {format_device(dev)}; tried: {format_capture_candidates(dev)}"
    )


def create_charuco_board(squares_x, squares_y, square_size, marker_size):
    """Create a Charuco board, ArUco dictionary, and ArUco detector."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_size, marker_size, aruco_dict)
    board.setLegacyPattern(True)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    return board, aruco_dict, aruco_detector


def split_stereo_frame(frame):
    """Split a side-by-side stereo frame into equal left/right images."""
    if frame is None or frame.size == 0:
        raise ValueError("empty frame")

    frame_h, frame_w = frame.shape[:2]
    half_w = frame_w // 2
    if half_w == 0:
        raise ValueError(f"invalid frame width: {frame_w}")

    left = frame[:, :half_w]
    right = frame[:, frame_w - half_w:]
    return left, right, (frame_w, frame_h), (half_w, frame_h)


def is_markers_found(frame, aruco_dict, squares_x, squares_y, min_detected_pct):
    """Check if enough ArUco markers are visible (percentage-based).

    Returns True if detected marker count >= floor(squaresX * squaresY / 2) * min_detected_pct.
    """
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    marker_corners, _, _ = detector.detectMarkers(frame)
    num_all_markers = math.floor(squares_x * squares_y / 2)
    min_needed = int(num_all_markers * min_detected_pct)
    found = len(marker_corners)
    return found >= min_needed, found, min_needed


def detect_charuco(frame_gray, board, aruco_detector):
    """Detect Charuco corners using 3-step pipeline:
    detectMarkers -> refineDetectedMarkers -> interpolateCornersCharuco.

    Returns (charuco_corners, charuco_ids, marker_corners, marker_ids)
    or (None, None, None, None) if detection fails.
    """
    # Step 1: detect ArUco markers
    marker_corners, marker_ids, rejected = aruco_detector.detectMarkers(frame_gray)

    # Step 2: refine — recover rejected markers near expected positions
    marker_corners, marker_ids, _, _ = cv2.aruco.refineDetectedMarkers(
        frame_gray, board, marker_corners, marker_ids,
        rejectedCorners=rejected)

    if marker_ids is None or len(marker_corners) == 0:
        return None, None, None, None

    # Step 3: interpolate Charuco corners from detected markers
    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, frame_gray, board, minMarkers=1)

    if ret > 0 and charuco_corners is not None:
        return charuco_corners, charuco_ids, marker_corners, marker_ids
    return None, None, None, None


def draw_charuco(frame, charuco_corners, charuco_ids):
    """Draw detected Charuco corners on frame (in-place)."""
    if charuco_corners is not None:
        cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, (0, 255, 0))
    return frame


def test_camera_orientation(frame_l, frame_r, aruco_dict):
    """Check that left camera is actually on the left.

    Compares x-coordinates of the same markers in both views.
    For a correctly oriented stereo pair, objects should appear further
    right in the left image than in the right image.
    Returns True if orientation is correct.
    """
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    marker_corners_l, id_l, _ = detector.detectMarkers(frame_l)
    marker_corners_r, id_r, _ = detector.detectMarkers(frame_r)

    if id_l is None or id_r is None:
        return True  # Can't verify, assume OK

    for i, left_id in enumerate(id_l):
        idx = np.where(id_r == left_id)
        if idx[0].size == 0:
            continue
        for left_corner, right_corner in zip(marker_corners_l[i], marker_corners_r[idx[0][0]]):
            if left_corner[0][0] - right_corner[0][0] < 0:
                return False
    return True


def capture_images(cap, board, aruco_dict, aruco_detector, args):
    """Interactive image capture loop with percentage-based filtering."""
    dataset_dir = Path(args.dataset)
    left_dir = dataset_dir / "left"
    right_dir = dataset_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    num_all_markers = math.floor(args.board_cols * args.board_rows / 2)
    min_markers = int(num_all_markers * args.minDetectedMarkersPercent)
    orientation_checked = False

    captured = 0
    actual_pair_size = None
    print("\n=== Image Capture ===")
    print(f"Target: {args.count} pairs | Board: Charuco {args.board_cols}x{args.board_rows}")
    print(f"Square: {args.square_size}cm | Marker: {args.marker_size}cm")
    print(f"Marker threshold: {min_markers}/{num_all_markers} ({args.minDetectedMarkersPercent:.0%})")
    print("[SPACE] capture | [ESC/q] finish\n")

    while captured < args.count:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break

        try:
            left, right, actual_frame_size, actual_pair_size = split_stereo_frame(frame)
        except ValueError as exc:
            print(f"[ERROR] Invalid camera frame: {exc}")
            break

        if actual_frame_size != (args.width, args.height):
            print(f"[WARN] Using actual frame size {actual_frame_size[0]}x{actual_frame_size[1]} for split")
            args.width, args.height = actual_frame_size

        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Step 1: percentage-based marker check
        l_ok, l_found, _ = is_markers_found(
            left_gray, aruco_dict, args.board_cols, args.board_rows,
            args.minDetectedMarkersPercent)
        r_ok, r_found, _ = is_markers_found(
            right_gray, aruco_dict, args.board_cols, args.board_rows,
            args.minDetectedMarkersPercent)

        # Step 2: full charuco detection with refine
        lc, li, _, _ = detect_charuco(left_gray, board, aruco_detector)
        rc, ri, _, _ = detect_charuco(right_gray, board, aruco_detector)

        capture_ok = l_ok and r_ok
        charuco_ok = lc is not None and rc is not None
        l_corners = len(lc) if lc is not None else 0
        r_corners = len(rc) if rc is not None else 0

        # Draw on display copies
        left_disp = left.copy()
        right_disp = right.copy()
        draw_charuco(left_disp, lc, li)
        draw_charuco(right_disp, rc, ri)

        # Status indicator with marker count
        l_color = (0, 255, 0) if l_ok else (0, 0, 255)
        r_color = (0, 255, 0) if r_ok else (0, 0, 255)
        cv2.putText(left_disp, f"L: {l_found}/{min_markers} markers", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, l_color, 2)
        cv2.putText(right_disp, f"R: {r_found}/{min_markers} markers", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, r_color, 2)

        if capture_ok and charuco_ok:
            status_color = (0, 255, 0)
            status_text = "READY"
        elif capture_ok:
            status_color = (0, 255, 255)
            status_text = "CAPTURE OK"
        else:
            status_color = (0, 0, 255)
            status_text = "NOT READY"
        cv2.putText(left_disp, f"[{status_text}] Captured: {captured}/{args.count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(left_disp, f"L: {l_corners} corners", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if lc is not None else (0, 255, 255), 2)
        cv2.putText(right_disp, f"R: {r_corners} corners", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if rc is not None else (0, 255, 255), 2)

        combined = np.hstack([left_disp, right_disp])
        scale = min(1.0, 1920.0 / combined.shape[1])
        if scale < 1.0:
            display = cv2.resize(combined, None, fx=scale, fy=scale)
        else:
            display = combined
        cv2.imshow("Stereo Calibration - [SPACE] capture, [ESC] quit", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord(' ') and capture_ok:
            # Orientation check on first capture
            if not orientation_checked:
                if not test_camera_orientation(left_gray, right_gray, aruco_dict):
                    print("  [ERROR] Camera orientation incorrect! Left/right may be swapped.")
                    print("  Please check camera mounting and restart.")
                    cv2.destroyAllWindows()
                    sys.exit(1)
                orientation_checked = True
                print("  [OK] Camera orientation verified")

            fname = f"img_{captured:03d}.png"
            cv2.imwrite(str(left_dir / fname), left)
            cv2.imwrite(str(right_dir / fname), right)
            captured += 1
            note = "" if charuco_ok else " [WARN: charuco interpolation failed in preview]"
            print(f"  [{captured}/{args.count}] Captured {fname} "
                  f"(L:{l_found} markers/{l_corners} corners, "
                  f"R:{r_found} markers/{r_corners} corners){note}")
            # Brief green flash
            flash = np.full_like(display, (0, 80, 0), dtype=np.uint8)
            cv2.imshow("Stereo Calibration - [SPACE] capture, [ESC] quit",
                       cv2.add(display, flash))
            cv2.waitKey(200)
        elif key == ord(' ') and not capture_ok:
            print(f"  [SKIP] Markers insufficient (L:{l_found} R:{r_found}, need {min_markers})")

    cv2.destroyAllWindows()
    if actual_pair_size is not None:
        print(f"Captured single-view image size: {actual_pair_size[0]}x{actual_pair_size[1]}")
    print(f"\nCapture complete: {captured} pairs saved to {dataset_dir.resolve()}/")
    return captured


def load_captured_images(dataset_dir):
    """Load captured image pairs from dataset directory."""
    left_dir = Path(dataset_dir) / "left"
    right_dir = Path(dataset_dir) / "right"

    left_files = sorted(left_dir.glob("*.png"))
    pairs = []
    for lf in left_files:
        rf = right_dir / lf.name
        if rf.exists():
            left_img = cv2.imread(str(lf), cv2.IMREAD_GRAYSCALE)
            right_img = cv2.imread(str(rf), cv2.IMREAD_GRAYSCALE)
            if left_img is not None and right_img is not None:
                pairs.append((left_img, right_img))
    print(f"Loaded {len(pairs)} image pairs from {Path(dataset_dir).resolve()}/")
    return pairs


def process_calibration(pairs, board, aruco_dict, aruco_detector, args):
    """Run stereo calibration on captured image pairs.

    Returns calibration result dict or None on failure.
    """
    print("\n=== Processing Calibration ===")
    first_left, _first_right = pairs[0]
    img_size = (first_left.shape[1], first_left.shape[0])

    num_all_markers = math.floor(args.board_cols * args.board_rows / 2)
    min_markers = int(num_all_markers * args.minDetectedMarkersPercent)

    all_obj_points = []
    all_left_corners = []
    all_right_corners = []

    for i, (left_gray, right_gray) in enumerate(pairs):
        # Percentage-based marker gate
        l_ok, l_found, _ = is_markers_found(
            left_gray, aruco_dict, args.board_cols, args.board_rows,
            args.minDetectedMarkersPercent)
        r_ok, r_found, _ = is_markers_found(
            right_gray, aruco_dict, args.board_cols, args.board_rows,
            args.minDetectedMarkersPercent)
        if not l_ok or not r_ok:
            print(f"  [SKIP] Pair {i}: markers insufficient "
                  f"(L:{l_found} R:{r_found}, need {min_markers})")
            continue

        # Full detection with refine
        lc, li, _, _ = detect_charuco(left_gray, board, aruco_detector)
        rc, ri, _, _ = detect_charuco(right_gray, board, aruco_detector)
        if lc is None or rc is None:
            print(f"  [SKIP] Pair {i}: charuco interpolation failed")
            continue

        # Find common charuco IDs
        left_id_set = set(li.flatten())
        right_id_set = set(ri.flatten())
        common_ids = sorted(left_id_set & right_id_set)
        if len(common_ids) < 6:
            print(f"  [SKIP] Pair {i}: only {len(common_ids)} common IDs")
            continue

        # Build matched arrays
        left_corners_matched = []
        right_corners_matched = []
        obj_points_matched = []
        board_corners_3d = board.getChessboardCorners()

        for cid in common_ids:
            l_idx = np.where(li.flatten() == cid)[0][0]
            r_idx = np.where(ri.flatten() == cid)[0][0]
            left_corners_matched.append(lc[l_idx])
            right_corners_matched.append(rc[r_idx])
            obj_points_matched.append(board_corners_3d[cid])

        obj_pts = np.array(obj_points_matched, dtype=np.float32)
        left_pts = np.array(left_corners_matched, dtype=np.float32)
        right_pts = np.array(right_corners_matched, dtype=np.float32)

        all_obj_points.append(obj_pts)
        all_left_corners.append(left_pts)
        all_right_corners.append(right_pts)
        print(f"  [OK] Pair {i}: {len(common_ids)} common corners")

    if len(all_obj_points) < 5:
        print(f"[ERROR] Only {len(all_obj_points)} valid pairs, need at least 5")
        return None

    print(f"\nUsing {len(all_obj_points)} valid pairs for calibration")

    # --- Individual camera calibration ---
    calib_flags = cv2.CALIB_RATIONAL_MODEL  # 14 distortion coefficients
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    print("Calibrating left camera...")
    ret_l, K_l, D_l, _rvecs_l, _tvecs_l = cv2.calibrateCamera(
        all_obj_points, all_left_corners, img_size, None, None,
        flags=calib_flags, criteria=criteria)
    print(f"  Left reprojection error: {ret_l:.6f}")

    print("Calibrating right camera...")
    ret_r, K_r, D_r, _rvecs_r, _tvecs_r = cv2.calibrateCamera(
        all_obj_points, all_right_corners, img_size, None, None,
        flags=calib_flags, criteria=criteria)
    print(f"  Right reprojection error: {ret_r:.6f}")

    # --- Stereo calibration ---
    print("Running stereo calibration...")
    stereo_flags = (cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_RATIONAL_MODEL)
    ret_s, K_l, D_l, K_r, D_r, R, T, _E, _F = cv2.stereoCalibrate(
        all_obj_points, all_left_corners, all_right_corners,
        K_l, D_l, K_r, D_r, img_size,
        flags=stereo_flags, criteria=criteria)
    print(f"  Stereo reprojection error: {ret_s:.6f}")

    # --- Stereo rectification ---
    print("Computing stereo rectification...")
    R1, R2, P1, P2, _Q, _roi1, _roi2 = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    # --- Show rectification preview ---
    print("\nShowing rectification preview (press any key to continue)...")
    show_rectification_preview(pairs, K_l, D_l, K_r, D_r, R1, R2, P1, P2, img_size)

    return {
        "K_l": K_l, "D_l": D_l, "ret_l": ret_l,
        "K_r": K_r, "D_r": D_r, "ret_r": ret_r,
        "R": R, "T": T, "ret_s": ret_s,
        "R1": R1, "R2": R2,
        "img_size": img_size,
    }


def show_rectification_preview(pairs, K_l, D_l, K_r, D_r, R1, R2, P1, P2, img_size):
    """Display rectified stereo pair with epipolar lines for verification."""
    map_l1, map_l2 = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, img_size, cv2.CV_32FC1)
    map_r1, map_r2 = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, img_size, cv2.CV_32FC1)

    # Pick a middle pair for preview
    idx = len(pairs) // 2
    left_gray, right_gray = pairs[idx]
    left_rect = cv2.remap(left_gray, map_l1, map_l2, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_gray, map_r1, map_r2, cv2.INTER_LINEAR)

    left_color = cv2.cvtColor(left_rect, cv2.COLOR_GRAY2BGR)
    right_color = cv2.cvtColor(right_rect, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([left_color, right_color])

    # Draw horizontal epipolar lines
    for y in range(0, combined.shape[0], 40):
        cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

    scale = min(1.0, 1920.0 / combined.shape[1])
    if scale < 1.0:
        combined = cv2.resize(combined, None, fx=scale, fy=scale)

    cv2.imshow("Rectification Preview - press any key", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def build_output_json(calib_result):
    """Build the output JSON dict matching calib_example.json format."""
    K_l = calib_result["K_l"]
    K_r = calib_result["K_r"]
    D_l = calib_result["D_l"]
    D_r = calib_result["D_r"]
    R = calib_result["R"]
    T = calib_result["T"]
    R1 = calib_result["R1"]
    R2 = calib_result["R2"]
    w, h = calib_result["img_size"]

    output = {
        "cameraData": [
            {
                "name": "left",
                "width": w,
                "height": h,
                "intrinsicMatrix": K_l.tolist(),
                "dist_coeff": D_l.flatten().tolist(),
                "reprojection_error": calib_result["ret_l"],
                "extrinsics": {
                    "to_cam": "right",
                    "rotationMatrix": R.tolist(),
                    "translation": T.flatten().tolist(),
                },
                "rectifiedRotation": R1.flatten().tolist(),
            },
            {
                "name": "right",
                "width": w,
                "height": h,
                "intrinsicMatrix": K_r.tolist(),
                "dist_coeff": D_r.flatten().tolist(),
                "reprojection_error": calib_result["ret_r"],
                "rectifiedRotation": R2.flatten().tolist(),
            },
        ]
    }
    return output


def main():
    args = parse_args()
    sdk = UVCStereo()
    dev = None

    board, aruco_dict, aruco_detector = create_charuco_board(
        args.board_cols, args.board_rows, args.square_size, args.marker_size)

    # --- Capture ---
    if "capture" in args.mode:
        dev = select_device(
            sdk,
            require_capture_source=True,
            device_index=args.device_index,
        )
        print(f"[INFO] Available capture sources: {format_capture_candidates(dev)}")
        cap = open_camera(dev, args.width, args.height)
        count = capture_images(cap, board, aruco_dict, aruco_detector, args)
        cap.release()
        if count < 5:
            print("[ERROR] Not enough images captured, need at least 5")
            sys.exit(1)

    # --- Process ---
    if "process" in args.mode:
        pairs = load_captured_images(args.dataset)
        if len(pairs) < 5:
            print("[ERROR] Not enough image pairs found, need at least 5")
            sys.exit(1)

        result = process_calibration(pairs, board, aruco_dict, aruco_detector, args)
        if result is None:
            print("[ERROR] Calibration failed")
            sys.exit(1)

        # Save JSON
        output = build_output_json(result)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)
        print(f"\n[OK] Calibration saved to {output_path.resolve()}")
        print(f"  Left  reprojection error: {result['ret_l']:.6f}")
        print(f"  Right reprojection error: {result['ret_r']:.6f}")
        print(f"  Stereo reprojection error: {result['ret_s']:.6f}")

        if dev is None:
            dev = select_device(
                sdk,
                require_capture_source=False,
                device_index=args.device_index,
            )
        json_text = json.dumps(output, indent=4)
        sdk.write_json(json_text, device=dev)
        print(f"\n[OK] Calibration written to device: {format_device(dev)}")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ERROR] Interrupted by user")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
