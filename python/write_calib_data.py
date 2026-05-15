from arducam_uvc_stereo_sdk import open_device, scan_devices
# Scan for devices
devices = scan_devices()
if not devices:
    raise RuntimeError("no devices found")

dev = devices[0]
print(
    f"selected device: vid=0x{dev.vid:04x} pid=0x{dev.pid:04x} "
    f"node={dev.video_node} bus={dev.bus_number} address={dev.device_address}"
)

# Load calibration JSON from a local file
with open("../calib_example.json", "r", encoding="utf-8") as f:
    json_text = f.read()

# Write calibration data to the device

camera = open_device(dev)
camera.write_json(json_text)
print("write_json success")

# Read back to verify
version, read_json = camera.read_json()
print(f"read version={version}")
print(f"read json={read_json}")
