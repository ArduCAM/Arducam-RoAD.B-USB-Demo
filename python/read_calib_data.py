from arducam_uvc_stereo_sdk import UVCStereo


sdk = UVCStereo()

# Scan for devices
devices = sdk.scan()
if not devices:
    raise RuntimeError("no devices found")

# Print device list
for i, dev in enumerate(devices):
    print(
        f"device[{i}]: vid=0x{dev.vid:04x} pid=0x{dev.pid:04x} "
        f"node={dev.video_node} bus={dev.bus_number} address={dev.device_address}"
    )

# Select the first device
dev = devices[0]

# Read calibration data from the device
version, json_text = sdk.read_json(device=dev)
print(f"version={version}")
print(f"json={json_text}")
