#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Any


from arducam_uvc_stereo_sdk import ImuConversionOptions, open_device, scan_devices
from utils import Axis3DViewer, OrientationEstimator, format_device_info, format_sample_line, sample_from_sdk


DEFAULT_INTERVAL_MS = 5.0
DEFAULT_ACCEL_BLEND = 0.02


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arducam Stereo IMU 3D axis viewer.")
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
        help=f"Accelerometer correction blend for roll/pitch. Default: {DEFAULT_ACCEL_BLEND:g}.",
    )
    parser.add_argument(
        "--gyro-only",
        action="store_true",
        help="Disable accelerometer correction and use gyroscope integration only.",
    )
    parser.add_argument(
        "--print-samples",
        action="store_true",
        help="Print converted IMU samples and estimated pose to stdout.",
    )
    return parser.parse_args(argv)


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


def selected_device() -> Any:
    devices = scan_devices()
    if not devices:
        raise RuntimeError("no Arducam UVC stereo devices found")
    return devices[0]


def close_imu(device: Any) -> None:
    close = getattr(device, "close_imu", None)
    if close is not None:
        try:
            close()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stop_requested = False

    def request_stop(signum: int, frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        device_info = selected_device()
        print(f"selected device: {format_device_info(device_info)}")
        imu = open_device(device_info)
        imu.open_imu()
    except Exception as exc:
        print(f"IMU startup failed: {exc}", file=sys.stderr)
        return 2

    viewer: Axis3DViewer | None = None
    try:
        conversion_options = ImuConversionOptions()
        estimator = OrientationEstimator(
            accel_blend=args.accel_blend,
            use_accel=not args.gyro_only,
        )
        viewer = Axis3DViewer("Arducam Stereo IMU 3D Axis")

        last_read_done: float | None = None
        while not stop_requested and viewer.tick():
            loop_start = time.monotonic()
            try:
                raw = imu.read_imu()
            except Exception as exc:
                print(f"IMU read failed: {exc}", file=sys.stderr)
                return 3

            read_done = time.monotonic()
            read_ms = (read_done - loop_start) * 1000.0
            interval_ms = 0.0 if last_read_done is None else (read_done - last_read_done) * 1000.0
            last_read_done = read_done

            try:
                sample = sample_from_sdk(raw, conversion_options)
            except Exception as exc:
                print(f"IMU conversion failed: {exc}", file=sys.stderr)
                return 4

            dt_s = interval_ms / 1000.0 if interval_ms > 0.0 else None
            pose = estimator.update(sample, dt_s=dt_s)

            if args.print_samples:
                print(
                    format_sample_line(
                        sample,
                        pose.roll_deg,
                        pose.pitch_deg,
                        pose.yaw_deg,
                        read_ms,
                        interval_ms,
                    ),
                    flush=True,
                )

            if not viewer.update_frame(pose, sample, read_ms, interval_ms):
                break

            sleep_until = loop_start + args.interval_ms / 1000.0
            while not stop_requested and time.monotonic() < sleep_until:
                if not viewer.tick():
                    break
                time.sleep(min(0.005, max(0.0, sleep_until - time.monotonic())))

    finally:
        close_imu(imu)
        if viewer is not None:
            viewer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
