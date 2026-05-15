from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ImuSample:
    temperature_c: float
    ax_g: float
    ay_g: float
    az_g: float
    gx_dps: float
    gy_dps: float
    gz_dps: float


def make_conversion_options() -> Any:
    from arducam_uvc_stereo_sdk import ImuConversionOptions

    return ImuConversionOptions()


def sample_from_sdk(raw: Any, options: Any | None = None) -> ImuSample:
    from arducam_uvc_stereo_sdk import convert_imu

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


def close_imu(device: Any) -> None:
    close = getattr(device, "close_imu", None)
    if close is not None:
        try:
            close()
        except Exception:
            pass
