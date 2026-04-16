#!/usr/bin/env python3
import sys

import cv2
import numpy as np
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from utils import (
    compute_maps,
    extract_stereo_params,
    get_capture_candidates,
    open_camera,
    read_device_calibration,
    select_device,
    split_stereo_frame,
)

PREVIEW_WINDOW_SIZES = {}


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
        show_preview(window_name, combined)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break


def main():
    cap = None

    try:
        configure_windows_dpi_awareness()
        from arducam_uvc_stereo_sdk import UVCStereo

        sdk = UVCStereo()
        dev = select_device(sdk, require_capture_source=True)
        calibration = read_device_calibration(sdk, dev)
        params = extract_stereo_params(calibration)
        maps = compute_maps(params)
        candidates = get_capture_candidates(dev)

        if not candidates:
            raise RuntimeError("selected device does not expose a usable capture source")

        print("available capture sources: " + ", ".join(candidate["label"] for candidate in candidates))
        cap = open_camera(candidates, *params["img_size"])
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
