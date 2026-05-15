from arducam_uvc_stereo_sdk import open_device, scan_devices


# Scan for devices
devices = scan_devices()
if not devices:
    raise RuntimeError("no devices found")

# Print device list
for i, dev in enumerate(devices):
    print(
        f"device[{i}]: vid=0x{dev.vid:04x} pid=0x{dev.pid:04x} "
        f"node={dev.video_node} bus={dev.bus_number} address={dev.device_address}, opencv={dev.opencv}"
    )
# Select the first device
dev = devices[0]

# Read calibration data from the device
camera = open_device(dev)
version, json_text = camera.read_json()
print(f"version={version}")
print(f"json={json_text}")
