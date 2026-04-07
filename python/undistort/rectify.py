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
    cap = None

    try:
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
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
