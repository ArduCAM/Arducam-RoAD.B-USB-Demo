from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arducam_uvc_stereo_sdk import ImuConversionOptions, convert_imu


@dataclass(frozen=True)
class ImuSample:
    temperature_c: float
    ax_g: float
    ay_g: float
    az_g: float
    gx_dps: float
    gy_dps: float
    gz_dps: float


def sample_from_sdk(raw: Any, options: ImuConversionOptions | None = None) -> ImuSample:
    converted = convert_imu(raw, options)
    return ImuSample(
        temperature_c=float(converted.temperature_c),
        ax_g=float(converted.accel_x_g),
        ay_g=float(converted.accel_y_g),
        az_g=float(converted.accel_z_g),
        gx_dps=float(converted.gyro_x_dps),
        gy_dps=float(converted.gyro_y_dps),
        gz_dps=float(converted.gyro_z_dps),
    )


def format_device_info(device: Any) -> str:
    return (
        f"vid=0x{int(device.vid):04x} "
        f"pid=0x{int(device.pid):04x} "
        f"node={getattr(device, 'video_node', '')} "
        f"bus={int(device.bus_number)} "
        f"address={int(device.device_address)}"
    )


def format_sample_line(
    sample: ImuSample,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    read_ms: float,
    interval_ms: float,
) -> str:
    return (
        "imu "
        f"temp={sample.temperature_c:.2f} C "
        f"accel=({sample.ax_g:.5f}, {sample.ay_g:.5f}, {sample.az_g:.5f}) g "
        f"gyro=({sample.gx_dps:.3f}, {sample.gy_dps:.3f}, {sample.gz_dps:.3f}) dps "
        f"pose=({roll_deg:.2f}, {pitch_deg:.2f}, {yaw_deg:.2f}) "
        f"read={read_ms:.3f} ms interval={interval_ms:.3f} ms"
    )
