#!/usr/bin/env python3
"""
Arducam UVC Mono Camera Calibration Tool

Captures images from one UVC camera, performs Charuco-based mono calibration,
outputs a flat mono calibration JSON, and writes the generated JSON to camera
flash.

Usage:
    python calibration/mono_calib.py
    python calibration/mono_calib.py -s 2.5 -ms 1.8 -nx 11 -ny 8
    python calibration/mono_calib.py --device-index 1
    python calibration/mono_calib.py -m process --dataset dataset/20260706_120000
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from arducam_uvc_stereo_sdk import open_device

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from calibration.stereo_calib import (  # noqa: E402
    configure_windows_dpi_awareness,
    create_charuco_board,
    destroy_preview_windows,
    detect_charuco,
    draw_charuco,
    is_markers_found,
    show_preview,
)
from utils import (  # noqa: E402
    format_device,
    get_capture_candidates,
    select_device,
)


def get_default_session_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PYTHON_DIR / "dataset" / timestamp


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Arducam UVC Mono Camera Charuco Calibration"
    )
    default_session_dir = get_default_session_dir()
    parser.add_argument("-W", "--width", type=int, default=1280,
                        help="Frame width. Default: 1280")
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
                        help="Target number of images. Default: 20")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output JSON path. Default: <dataset>/calib_result.json")
    parser.add_argument("-m", "--mode", type=str, default="capture+process",
                        choices=["capture+process", "capture", "process"],
                        help="Mode: capture+process, capture only, or process only. Default: capture+process")
    parser.add_argument("--dataset", type=str, default=str(default_session_dir),
                        help="Directory to save captured images and calibration result. Default: <project_root>/dataset/<timestamp>")
    parser.add_argument("-mdmp", "--minDetectedMarkersPercent", type=float, default=0.4,
                        help="Min percentage of detected markers to accept a frame. Default: 0.4")
    parser.add_argument("--device-index", type=int, default=None,
                        help="Index in scan_devices() result. Default: prompt in CLI when multiple devices are detected")

    args = parser.parse_args(argv)
    if args.output is None:
        args.output = str(Path(args.dataset) / "calib_result.json")

    validate_args(parser, args)
    return args


def validate_args(parser, args):
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.board_cols <= 1 or args.board_rows <= 1:
        parser.error("--board-cols and --board-rows must be greater than 1")
    if args.square_size <= 0 or args.marker_size <= 0:
        parser.error("--square-size and --marker-size must be positive")
    if args.marker_size >= args.square_size:
        parser.error("--marker-size must be smaller than --square-size")
    if args.count <= 0:
        parser.error("--count must be positive")
    if not 0 < args.minDetectedMarkersPercent <= 1:
        parser.error("--minDetectedMarkersPercent must be in the range (0, 1]")


def format_capture_candidates(dev):
    candidates = get_capture_candidates(dev)
    if not candidates:
        return "none"
    return ", ".join(candidate["label"] for candidate in candidates)


def open_mono_camera(candidates, width, height):
    if not candidates:
        raise RuntimeError("no usable capture source")

    errors = []
    for candidate in candidates:
        label = candidate["label"]
        print(f"trying camera source: {label}")
        time.sleep(0.1)
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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"camera opened via {label}: {actual_w}x{actual_h} MJPG")

        if actual_w != width or actual_h != height:
            cap.release()
            errors.append(f"{label}: requested {width}x{height}, got {actual_w}x{actual_h}")
            print(
                f"size mismatch via {label}: requested {width}x{height}, "
                f"got {actual_w}x{actual_h}"
            )
            continue

        return cap

    raise RuntimeError("cannot open camera from candidates: " + ", ".join(errors))


def to_gray(frame):
    if frame is None or frame.size == 0:
        raise ValueError("empty frame")
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"unsupported frame shape: {frame.shape}")


def to_bgr(frame):
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame.copy()
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"unsupported frame shape: {frame.shape}")


def capture_images(cap, board, aruco_dict, aruco_detector, args):
    dataset_dir = Path(args.dataset)
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    num_all_markers = math.floor(args.board_cols * args.board_rows / 2)
    min_markers = int(num_all_markers * args.minDetectedMarkersPercent)

    captured = 0
    actual_size = None
    print("\n=== Image Capture ===")
    print(f"Target: {args.count} images | Board: Charuco {args.board_cols}x{args.board_rows}")
    print(f"Square: {args.square_size}cm | Marker: {args.marker_size}cm")
    print(f"Marker threshold: {min_markers}/{num_all_markers} ({args.minDetectedMarkersPercent:.0%})")
    print("[SPACE] capture | [ESC/q] finish\n")
    capture_window = "Mono Calibration - [SPACE] capture, [ESC] quit"

    while captured < args.count:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break

        try:
            frame_gray = to_gray(frame)
            frame_disp = to_bgr(frame)
        except ValueError as exc:
            print(f"[ERROR] Invalid camera frame: {exc}")
            break

        frame_h, frame_w = frame_gray.shape[:2]
        actual_size = (frame_w, frame_h)
        if actual_size != (args.width, args.height):
            print(f"[WARN] Using actual frame size {frame_w}x{frame_h}")
            args.width, args.height = actual_size

        marker_ok, found, _ = is_markers_found(
            frame_gray, aruco_dict, args.board_cols, args.board_rows,
            args.minDetectedMarkersPercent)
        charuco_corners, charuco_ids, _, _ = detect_charuco(
            frame_gray, board, aruco_detector)

        draw_charuco(frame_disp, charuco_corners, charuco_ids)
        corners_count = len(charuco_corners) if charuco_corners is not None else 0
        marker_color = (0, 255, 0) if marker_ok else (0, 0, 255)
        status_color = (0, 255, 0) if marker_ok else (0, 0, 255)
        status_text = "READY" if marker_ok else "NOT READY"

        cv2.putText(frame_disp, f"Markers: {found}/{min_markers}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, marker_color, 2)
        cv2.putText(frame_disp, f"[{status_text}] Captured: {captured}/{args.count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame_disp, f"Corners: {corners_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if charuco_corners is not None else (0, 255, 255), 2)

        display = show_preview(capture_window, frame_disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
        if key == ord(" ") and marker_ok:
            fname = f"img_{captured:03d}.png"
            output_path = images_dir / fname
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"failed to write image: {output_path}")
            captured += 1
            note = "" if charuco_corners is not None else " [WARN: charuco interpolation failed in preview]"
            print(f"  [{captured}/{args.count}] Captured {fname} "
                  f"({found} markers/{corners_count} corners){note}")
            flash = np.full_like(display, (0, 80, 0), dtype=np.uint8)
            show_preview(capture_window, cv2.add(display, flash))
            cv2.waitKey(200)
        elif key == ord(" ") and not marker_ok:
            print(f"  [SKIP] Markers insufficient ({found}, need {min_markers})")

    destroy_preview_windows()
    if actual_size is not None:
        print(f"Captured image size: {actual_size[0]}x{actual_size[1]}")
    print(f"\nCapture complete: {captured} images saved to {images_dir.resolve()}/")
    return captured


def load_captured_images(dataset_dir):
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / "images"
    search_dir = images_dir if images_dir.is_dir() else dataset_path

    image_files = sorted(search_dir.glob("*.png"))
    images = []
    for image_file in image_files:
        image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  [SKIP] Failed to read image: {image_file}")
            continue
        images.append((image_file, image))

    print(f"Loaded {len(images)} images from {search_dir.resolve()}/")
    return images


def process_calibration(images, board, aruco_dict, aruco_detector, args):
    print("\n=== Processing Calibration ===")
    first_path, first_gray = images[0]
    img_size = (first_gray.shape[1], first_gray.shape[0])
    print(f"Reference image size: {img_size[0]}x{img_size[1]} ({first_path.name})")

    num_all_markers = math.floor(args.board_cols * args.board_rows / 2)
    min_markers = int(num_all_markers * args.minDetectedMarkersPercent)
    board_corners_3d = board.getChessboardCorners()

    all_obj_points = []
    all_image_corners = []

    for i, (image_file, image_gray) in enumerate(images):
        current_size = (image_gray.shape[1], image_gray.shape[0])
        if current_size != img_size:
            print(f"  [SKIP] Image {i} ({image_file.name}): size {current_size[0]}x{current_size[1]} "
                  f"does not match {img_size[0]}x{img_size[1]}")
            continue

        marker_ok, found, _ = is_markers_found(
            image_gray, aruco_dict, args.board_cols, args.board_rows,
            args.minDetectedMarkersPercent)
        if not marker_ok:
            print(f"  [SKIP] Image {i} ({image_file.name}): markers insufficient "
                  f"({found}, need {min_markers})")
            continue

        charuco_corners, charuco_ids, _, _ = detect_charuco(
            image_gray, board, aruco_detector)
        if charuco_corners is None or charuco_ids is None:
            print(f"  [SKIP] Image {i} ({image_file.name}): charuco interpolation failed")
            continue

        ids = charuco_ids.flatten()
        if len(ids) < 6:
            print(f"  [SKIP] Image {i} ({image_file.name}): only {len(ids)} charuco corners")
            continue

        obj_points = np.array(
            [board_corners_3d[int(corner_id)] for corner_id in ids],
            dtype=np.float32,
        )
        image_points = np.asarray(charuco_corners, dtype=np.float32)
        all_obj_points.append(obj_points)
        all_image_corners.append(image_points)
        print(f"  [OK] Image {i} ({image_file.name}): {len(ids)} corners")

    if len(all_obj_points) < 5:
        print(f"[ERROR] Only {len(all_obj_points)} valid images, need at least 5")
        return None

    print(f"\nUsing {len(all_obj_points)} valid images for calibration")
    calib_flags = cv2.CALIB_RATIONAL_MODEL
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    try:
        ret, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
            all_obj_points,
            all_image_corners,
            img_size,
            None,
            None,
            flags=calib_flags,
            criteria=criteria,
        )
    except cv2.error as exc:
        raise RuntimeError(f"mono calibration failed: {exc}") from exc

    print(f"  Reprojection error: {ret:.6f}")
    print("\nShowing undistortion preview (press any key to continue)...")
    show_undistortion_preview(images, camera_matrix, dist_coeffs, img_size)

    return {
        "K": camera_matrix,
        "D": dist_coeffs,
        "ret": ret,
        "img_size": img_size,
    }


def show_undistortion_preview(images, camera_matrix, dist_coeffs, img_size):
    idx = len(images) // 2
    _image_file, image_gray = images[idx]
    # Preserve the calibrated pixel scale.  Using getOptimalNewCameraMatrix()
    # here alters the focal length to crop borders, which scales the result.
    undistorted = cv2.undistort(
        image_gray, camera_matrix, dist_coeffs, None, camera_matrix)

    original_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    undistorted_color = cv2.cvtColor(undistorted, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([original_color, undistorted_color])

    cv2.putText(combined, "original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined, "undistorted", (img_size[0] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    show_preview("Undistortion Preview - press any key", combined)
    cv2.waitKey(0)
    destroy_preview_windows()


def build_output_json(calib_result):
    camera_matrix = calib_result["K"]
    dist_coeffs = calib_result["D"]
    width, height = calib_result["img_size"]

    return {
        "width": width,
        "height": height,
        "intrinsicMatrix": camera_matrix.tolist(),
        "dist_coeff": dist_coeffs.flatten().tolist(),
        "reprojection_error": calib_result["ret"],
    }


def write_calibration_to_device(dev, output):
    json_text = json.dumps(output, indent=4)
    try:
        camera = open_device(dev)
        camera.write_json(json_text)
        version, _read_json = camera.read_json()
    except Exception as exc:
        raise RuntimeError(
            f"failed to write calibration JSON to device {format_device(dev)}: {exc}"
        ) from exc

    print(f"\n[OK] Calibration written to device: {format_device(dev)}")
    print(f"[OK] Calibration read back from device, version={version}")
    return json_text


def main():
    configure_windows_dpi_awareness()
    args = parse_args()
    dev = None

    board, aruco_dict, aruco_detector = create_charuco_board(
        args.board_cols, args.board_rows, args.square_size, args.marker_size)

    if "capture" in args.mode:
        dev = select_device(
            require_capture_source=True,
            device_index=args.device_index,
        )
        print(f"[INFO] Available capture sources: {format_capture_candidates(dev)}")
        candidates = get_capture_candidates(dev)
        if not candidates:
            raise RuntimeError("selected device does not expose a usable capture source")
        cap = open_mono_camera(candidates, args.width, args.height)
        try:
            count = capture_images(cap, board, aruco_dict, aruco_detector, args)
        finally:
            cap.release()
        if count < 5:
            print("[ERROR] Not enough images captured, need at least 5")
            sys.exit(1)

    if "process" in args.mode:
        images = load_captured_images(args.dataset)
        if len(images) < 5:
            print("[ERROR] Not enough images found, need at least 5")
            sys.exit(1)

        result = process_calibration(images, board, aruco_dict, aruco_detector, args)
        if result is None:
            print("[ERROR] Calibration failed")
            sys.exit(1)

        output = build_output_json(result)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)
        print(f"\n[OK] Calibration saved to {output_path.resolve()}")
        print(f"  Reprojection error: {result['ret']:.6f}")

        if dev is None:
            dev = select_device(
                require_capture_source=False,
                device_index=args.device_index,
            )
        write_calibration_to_device(dev, output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ERROR] Interrupted by user")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
