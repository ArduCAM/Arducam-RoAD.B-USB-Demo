#!/usr/bin/env python3
import sys

import cv2
import numpy as np
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from utils import (
    get_capture_candidates,
    read_device_calibration,
    select_device,
)

PREVIEW_WINDOW_SIZES = {}


def extract_mono_params(calibration):
    """Accept a flat mono calibration JSON or a stereo cameraData payload."""

    def parse_camera(camera):
        if not isinstance(camera, dict):
            raise RuntimeError("calibration entry is not a dict")

        try:
            width = int(camera["width"])
            height = int(camera["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid width/height in calibration JSON: {exc}") from exc

        if width <= 0 or height <= 0:
            raise RuntimeError("width and height in calibration JSON must be positive")

        try:
            K = np.asarray(camera["intrinsicMatrix"], dtype=np.float64)
            D = np.asarray(camera["dist_coeff"], dtype=np.float64).reshape(-1, 1)
        except KeyError as exc:
            raise RuntimeError(f"missing calibration field: {exc}") from exc

        if K.shape != (3, 3):
            raise RuntimeError(f"intrinsicMatrix must have shape (3, 3), got {K.shape}")
        if D.size == 0:
            raise RuntimeError("dist_coeff is empty")

        return {"img_size": (width, height), "K": K, "D": D}

    camera_data = calibration.get("cameraData")
    if isinstance(camera_data, list) and camera_data:
        for preferred_name in ("left", "right"):
            for camera in camera_data:
                if isinstance(camera, dict) and camera.get("name") == preferred_name:
                    return parse_camera(camera)
        return parse_camera(camera_data[0])

    return parse_camera(calibration)


def compute_mono_maps(params):
    img_size = params["img_size"]
    map_x, map_y = cv2.initUndistortRectifyMap(
        params["K"],
        params["D"],
        None,
        params["K"],
        img_size,
        cv2.CV_32FC1,
    )
    return map_x, map_y


def configure_windows_dpi_awareness():
    """Enable Windows per-monitor DPI awareness before creating UI windows."""
    if not sys.platform.startswith("win"):
        return

    try:
        import ctypes

        user32 = ctypes.windll.user32
        pointer_bits = ctypes.sizeof(ctypes.c_void_p) * 8
        pmv2_context = (-4) & ((1 << pointer_bits) - 1)

        try:
            user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
            user32.SetProcessDpiAwarenessContext.restype = ctypes.c_bool
            if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(pmv2_context)):
                return
        except AttributeError:
            pass

        try:
            shcore = ctypes.windll.shcore
            shcore.SetProcessDpiAwareness.argtypes = [ctypes.c_int]
            shcore.SetProcessDpiAwareness.restype = ctypes.c_long
            shcore.SetProcessDpiAwareness(2)
            return
        except AttributeError:
            pass

        user32.SetProcessDPIAware()
    except Exception:
        # DPI awareness is best-effort. Preview scaling still works as fallback.
        pass


def _get_windows_work_area():
    """Return primary monitor work area in pixels on Windows."""
    if not sys.platform.startswith("win"):
        return None

    try:
        import ctypes
        from ctypes import wintypes

        rect = wintypes.RECT()
        spi_get_workarea = 0x0030
        if ctypes.windll.user32.SystemParametersInfoW(
                spi_get_workarea, 0, ctypes.byref(rect), 0):
            width = rect.right - rect.left
            height = rect.bottom - rect.top
            if width > 0 and height > 0:
                return width, height

        width = ctypes.windll.user32.GetSystemMetrics(0)
        height = ctypes.windll.user32.GetSystemMetrics(1)
        if width > 0 and height > 0:
            return width, height
    except Exception:
        return None

    return None


def get_preview_max_size():
    """Return a safe preview area that fits typical desktop work areas."""
    work_area = _get_windows_work_area()
    if work_area is not None:
        work_w, work_h = work_area
        return max(640, work_w - 80), max(480, work_h - 120)

    return 1920, 1080


def resize_for_preview(image):
    """Scale preview image down to fit the available desktop work area."""
    max_w, max_h = get_preview_max_size()
    height, width = image.shape[:2]
    scale = min(1.0, max_w / float(width), max_h / float(height))
    if scale >= 1.0:
        return image

    target_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def show_preview(window_name, image):
    """Show preview image in a manually sized window."""
    display = resize_for_preview(image)

    if window_name not in PREVIEW_WINDOW_SIZES:
        flags = cv2.WINDOW_NORMAL
        if hasattr(cv2, "WINDOW_KEEPRATIO"):
            flags |= cv2.WINDOW_KEEPRATIO
        cv2.namedWindow(window_name, flags)
        PREVIEW_WINDOW_SIZES[window_name] = None

    window_size = (display.shape[1], display.shape[0])
    if PREVIEW_WINDOW_SIZES[window_name] != window_size:
        cv2.resizeWindow(window_name, *window_size)
        PREVIEW_WINDOW_SIZES[window_name] = window_size

    cv2.imshow(window_name, display)


def destroy_preview_windows():
    """Close preview windows and reset tracked window sizes."""
    PREVIEW_WINDOW_SIZES.clear()
    cv2.destroyAllWindows()


def open_mono_camera(candidates, width, height):
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


def preview_loop(cap, maps, img_size):
    map_x, map_y = maps
    window_name = "Live Mono - [s] switch, [q]/[Esc] quit"
    show_rectified = True

    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame from camera")

        if frame is None or frame.size == 0:
            raise RuntimeError("empty frame from camera")

        rectified = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        display = rectified if show_rectified else frame
        show_preview(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            show_rectified = not show_rectified
        elif key in (27, ord("q")):
            break


def main():
    cap = None

    try:
        configure_windows_dpi_awareness()

        dev = select_device(require_capture_source=True)
        calibration = read_device_calibration(dev)
        params = extract_mono_params(calibration)
        maps = compute_mono_maps(params)
        candidates = get_capture_candidates(dev)

        if not candidates:
            raise RuntimeError("selected device does not expose a usable capture source")

        print("available capture sources: " + ", ".join(candidate["label"] for candidate in candidates))
        cap = open_mono_camera(candidates, *params["img_size"])
        preview_loop(cap, maps, params["img_size"])
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        if cap is not None:
            cap.release()
        destroy_preview_windows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
