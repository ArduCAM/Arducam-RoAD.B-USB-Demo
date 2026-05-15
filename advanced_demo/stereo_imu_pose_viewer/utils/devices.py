from __future__ import annotations

from typing import Any

import cv2


def format_device(dev: Any) -> str:
    parts = []
    vid = getattr(dev, "vid", None)
    pid = getattr(dev, "pid", None)
    video_node = getattr(dev, "video_node", None)
    bus_number = getattr(dev, "bus_number", None)
    device_address = getattr(dev, "device_address", None)

    if vid is not None:
        parts.append(f"vid=0x{int(vid):04x}")
    if pid is not None:
        parts.append(f"pid=0x{int(pid):04x}")
    if video_node:
        parts.append(f"node={video_node}")
    if bus_number is not None:
        parts.append(f"bus={int(bus_number)}")
    if device_address is not None:
        parts.append(f"address={int(device_address)}")

    return " ".join(parts) if parts else str(dev)


def format_device_vid_pid(dev: Any) -> str:
    vid = getattr(dev, "vid", None)
    pid = getattr(dev, "pid", None)
    vid_desc = f"0x{int(vid):04x}" if vid is not None else "unknown"
    pid_desc = f"0x{int(pid):04x}" if pid is not None else "unknown"

    return f"VID={vid_desc} PID={pid_desc}"


def _backend_name(backend: Any) -> str:
    return getattr(backend, "name", str(backend))


def get_opencv_candidates(dev: Any) -> list[tuple[str, int]]:
    candidates: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
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
                if key in seen or getattr(cv2, backend_name, None) is None:
                    continue
                seen.add(key)
                candidates.append(key)
        if candidates:
            return candidates

    backend_indices = getattr(dev, "opencv_backend_indices", None)
    if isinstance(backend_indices, dict):
        for backend, source_id in backend_indices.items():
            backend_name = _backend_name(backend)
            try:
                source_id = int(source_id)
            except (TypeError, ValueError):
                continue
            key = (backend_name, source_id)
            if key in seen or getattr(cv2, backend_name, None) is None:
                continue
            seen.add(key)
            candidates.append(key)

    return candidates


def get_capture_candidates(dev: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for backend_name, source_id in get_opencv_candidates(dev):
        backend_id = getattr(cv2, backend_name, None)
        if backend_id is None:
            continue
        key = ("opencv", backend_name, source_id)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "kind": "opencv",
                "backend_name": backend_name,
                "backend_id": int(backend_id),
                "source": source_id,
                "label": f"{backend_name}:{source_id}",
            }
        )

    video_node = getattr(dev, "video_node", None)
    if video_node:
        backend_id = getattr(cv2, "CAP_V4L2", None)
        key = ("video_node", video_node)
        if key not in seen:
            seen.add(key)
            candidates.append(
                {
                    "kind": "video_node",
                    "backend_name": "CAP_V4L2" if backend_id is not None else None,
                    "backend_id": int(backend_id) if backend_id is not None else None,
                    "source": video_node,
                    "label": f"video_node:{video_node}",
                }
            )

    return candidates


def build_device_entries(devices: list[Any]) -> list[dict[str, Any]]:
    entries = []
    for scan_index, dev in enumerate(devices):
        capture_candidates = get_capture_candidates(dev)
        capture_desc = (
            "none"
            if not capture_candidates
            else ", ".join(candidate["label"] for candidate in capture_candidates)
        )
        device_desc = format_device(dev)
        menu_desc = f"device[{scan_index}] {format_device_vid_pid(dev)}"
        entries.append(
            {
                "scan_index": scan_index,
                "dev": dev,
                "device_desc": device_desc,
                "capture_desc": capture_desc,
                "has_capture_source": bool(capture_candidates),
                "label": menu_desc,
            }
        )
    return entries


def resolve_device(devices: list[Any], scan_index: int) -> Any:
    if scan_index < 0 or scan_index >= len(devices):
        raise RuntimeError(
            f"device index {scan_index} is out of range for {len(devices)} detected device(s)"
        )
    return devices[scan_index]
