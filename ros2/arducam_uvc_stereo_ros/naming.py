"""Naming helpers for camera namespaces and frame prefixes."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence


def namespace_from_device_index(device_index: str) -> str:
    try:
        index = int(device_index)
    except ValueError:
        return ""
    if index < 0:
        return ""
    return f"cam{index}"


def camera_name_from_namespace(camera_name: str, namespace: str) -> str:
    if camera_name.strip():
        return camera_name.strip()
    namespace_name = namespace.strip().strip("/").replace("/", "_")
    return namespace_name or "stereo"


def auto_namespace_from_ros_args(args: Sequence[str]) -> str:
    if _has_namespace_remap(args):
        return ""

    device_index = _device_index_from_ros_args(args)
    if device_index is None:
        return ""
    return namespace_from_device_index(device_index)


def _device_index_from_ros_args(args: Sequence[str]) -> Optional[str]:
    device_index = None
    for assignment in _option_values(args, ("-p", "--param")):
        if ":=" not in assignment:
            continue
        name, value = assignment.split(":=", 1)
        if _assignment_targets(name, "device_index"):
            device_index = value.strip().strip("\"'")
    return device_index


def _has_namespace_remap(args: Sequence[str]) -> bool:
    for rule in _option_values(args, ("-r", "--remap")):
        if ":=" not in rule:
            continue
        source, _ = rule.split(":=", 1)
        source = source.strip()
        if source == "__ns" or source.endswith(":__ns"):
            return True
    return False


def _assignment_targets(name: str, target: str) -> bool:
    name = name.strip()
    return name == target or name.endswith(f".{target}") or name.endswith(f":{target}")


def _option_values(args: Sequence[str], options: Iterable[str]) -> Iterable[str]:
    option_set = tuple(options)
    index = 0
    while index < len(args):
        token = args[index]

        if token in option_set:
            if index + 1 < len(args):
                yield args[index + 1]
            index += 2
            continue

        for option in option_set:
            if token.startswith(f"{option}="):
                yield token[len(option) + 1 :]
                break
            if len(option) == 2 and token.startswith(option) and len(token) > 2:
                yield token[len(option) :]
                break

        index += 1
