"""CLI for listing connected Arducam UVC stereo devices."""

from __future__ import annotations

import sys

from arducam_uvc_stereo_sdk import UVCStereo

from .device_selector import format_device_table


def main() -> int:
    sdk = UVCStereo()
    devices = sdk.scan()
    if not devices:
        print("No Arducam UVC stereo devices found.")
        return 1

    print(format_device_table(devices))
    return 0


if __name__ == "__main__":
    sys.exit(main())
