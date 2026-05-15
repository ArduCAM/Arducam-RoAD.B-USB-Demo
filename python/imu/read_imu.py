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
print(
    f"selected device: vid=0x{dev.vid:04x} pid=0x{dev.pid:04x} "
    f"node={dev.video_node} bus={dev.bus_number} address={dev.device_address}"
)

# Open IMU
camera = open_device(dev)
camera.open_imu()

# Read IMU data
try:
    imu_data = camera.read_imu()
finally:
    camera.close_imu()

print(
    "imu data:\n"
    f"  temperature_raw: {imu_data.temperature_raw}\n"
    f"  accel_raw:       x={imu_data.accel_x_raw}, "
    f"y={imu_data.accel_y_raw}, "
    f"z={imu_data.accel_z_raw}\n"
    f"  gyro_raw:        x={imu_data.gyro_x_raw}, "
    f"y={imu_data.gyro_y_raw}, "
    f"z={imu_data.gyro_z_raw}"
)
