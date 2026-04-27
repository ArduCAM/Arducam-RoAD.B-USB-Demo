"""Device selection helpers for Arducam UVC stereo cameras."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


class SelectionError(RuntimeError):
    """Raised when device selection is invalid or ambiguous."""


@dataclass(frozen=True)
class SelectionParameters:
    """Normalized device selection parameters."""

    serial_number: str = ""
    video_node: str = ""
    bus_number: int = -1
    device_address: int = -1
    vid: int = -1
    pid: int = -1
    device_index: int = -1


@dataclass(frozen=True)
class SelectionResult:
    """Selection result including the active selection mode."""

    device: object
    mode: str
    warnings: Tuple[str, ...]


def _device_attr(device: object, name: str, default: object = "") -> object:
    return getattr(device, name, default)


def describe_device(device: object) -> str:
    """Return a readable description for logs and errors."""

    vid = int(_device_attr(device, "vid", 0))
    pid = int(_device_attr(device, "pid", 0))
    serial = str(_device_attr(device, "serial_number", "")) or "-"
    video_node = str(_device_attr(device, "video_node", "")) or "-"
    bus = _device_attr(device, "bus_number", "-")
    address = _device_attr(device, "device_address", "-")
    manufacturer = str(_device_attr(device, "manufacturer", "")) or "-"
    product = str(_device_attr(device, "product", "")) or "-"
    return (
        f"serial={serial} node={video_node} bus={bus} address={address} "
        f"vid=0x{vid:04x} pid=0x{pid:04x} manufacturer={manufacturer} product={product}"
    )


def format_device_table(devices: Iterable[object]) -> str:
    """Return a fixed-column table for device enumeration and error output."""

    rows = [
        "index  serial_number  video_node   bus  addr  vid     pid     manufacturer  product",
        "-----  -------------  ----------  ---  ----  ------  ------  ------------  -------",
    ]
    for index, device in enumerate(devices):
        rows.append(
            "{index:<5}  {serial:<13}  {video:<10}  {bus:<3}  {addr:<4}  "
            "0x{vid:04x}  0x{pid:04x}  {manufacturer:<12}  {product}".format(
                index=index,
                serial=str(_device_attr(device, "serial_number", "")) or "-",
                video=str(_device_attr(device, "video_node", "")) or "-",
                bus=str(_device_attr(device, "bus_number", "-")),
                addr=str(_device_attr(device, "device_address", "-")),
                vid=int(_device_attr(device, "vid", 0)),
                pid=int(_device_attr(device, "pid", 0)),
                manufacturer=str(_device_attr(device, "manufacturer", "")) or "-",
                product=str(_device_attr(device, "product", "")) or "-",
            )
        )
    return "\n".join(rows)


def select_device(devices: Sequence[object], params: SelectionParameters) -> SelectionResult:
    """Select one device according to the fixed priority rules."""

    if not devices:
        raise SelectionError("No Arducam UVC stereo devices were found.")

    warnings: List[str] = []

    if params.serial_number:
        warnings.extend(_ignored_warnings("serial_number", params))
        matches = [d for d in devices if str(_device_attr(d, "serial_number", "")) == params.serial_number]
        return SelectionResult(_require_single_match(matches, "serial_number", params.serial_number), "serial_number", tuple(warnings))

    if params.video_node:
        warnings.extend(_ignored_warnings("video_node", params))
        matches = [d for d in devices if str(_device_attr(d, "video_node", "")) == params.video_node]
        return SelectionResult(_require_single_match(matches, "video_node", params.video_node), "video_node", tuple(warnings))

    if params.bus_number != -1 or params.device_address != -1:
        if params.bus_number == -1 or params.device_address == -1:
            raise SelectionError("bus_number and device_address must both be provided together.")
        warnings.extend(_ignored_warnings("bus_address", params))
        matches = [
            d
            for d in devices
            if int(_device_attr(d, "bus_number", -1)) == params.bus_number
            and int(_device_attr(d, "device_address", -1)) == params.device_address
        ]
        return SelectionResult(
            _require_single_match(matches, "bus_number/device_address", f"{params.bus_number}/{params.device_address}"),
            "bus_address",
            tuple(warnings),
        )

    if params.vid != -1 or params.pid != -1 or params.device_index != -1:
        if params.vid != -1 or params.pid != -1:
            if params.vid == -1 or params.pid == -1 or params.device_index == -1:
                raise SelectionError("vid and pid require device_index, and device_index requires both vid and pid in this mode.")
            matches = [
                d
                for d in devices
                if int(_device_attr(d, "vid", -1)) == params.vid and int(_device_attr(d, "pid", -1)) == params.pid
            ]
            if not matches:
                raise SelectionError(
                    f"No devices matched vid=0x{params.vid:04x} pid=0x{params.pid:04x}."
                )
            if params.device_index < 0 or params.device_index >= len(matches):
                raise SelectionError(
                    f"device_index {params.device_index} is out of range for {len(matches)} device(s) matching "
                    f"vid=0x{params.vid:04x} pid=0x{params.pid:04x}."
                )
            return SelectionResult(matches[params.device_index], "vid_pid_device_index", tuple(warnings))

        if params.device_index < 0 or params.device_index >= len(devices):
            raise SelectionError(
                f"device_index {params.device_index} is out of range for {len(devices)} detected device(s)."
            )
        return SelectionResult(devices[params.device_index], "device_index", tuple(warnings))

    if len(devices) == 1:
        return SelectionResult(devices[0], "auto_single", tuple(warnings))

    raise SelectionError(
        "Multiple devices were detected and no unique selector was provided. "
        "Set serial_number or video_node to choose one camera."
    )


def _ignored_warnings(mode: str, params: SelectionParameters) -> List[str]:
    ignored = []
    if mode == "serial_number":
        if params.video_node:
            ignored.append("video_node")
        if params.bus_number != -1 or params.device_address != -1:
            ignored.append("bus_number/device_address")
        if params.vid != -1 or params.pid != -1:
            ignored.append("vid/pid")
        if params.device_index != -1:
            ignored.append("device_index")
    elif mode == "video_node":
        if params.bus_number != -1 or params.device_address != -1:
            ignored.append("bus_number/device_address")
        if params.vid != -1 or params.pid != -1:
            ignored.append("vid/pid")
        if params.device_index != -1:
            ignored.append("device_index")
    elif mode == "bus_address":
        if params.vid != -1 or params.pid != -1:
            ignored.append("vid/pid")
        if params.device_index != -1:
            ignored.append("device_index")

    if not ignored:
        return []
    return [f"Ignoring lower-priority selection parameters because {mode} is active: {', '.join(ignored)}."]


def _require_single_match(matches: Sequence[object], field_name: str, field_value: object) -> object:
    if not matches:
        raise SelectionError(f"No devices matched {field_name}={field_value}.")
    if len(matches) > 1:
        raise SelectionError(
            f"{len(matches)} devices matched {field_name}={field_value}; selection must resolve to exactly one device."
        )
    return matches[0]
