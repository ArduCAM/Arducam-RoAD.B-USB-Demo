from arducam_uvc_stereo_sdk import (
    ReportedDeviceCapability,
    open_device,
    scan_devices,
)

CAPABILITY_LABELS = (
    (ReportedDeviceCapability.VIDEO, "Video"),
    (ReportedDeviceCapability.FLASH, "Flash"),
    (ReportedDeviceCapability.IMU, "IMU"),
    (ReportedDeviceCapability.MICROPHONE, "Microphone"),
    (ReportedDeviceCapability.EXT_TRIGGER, "External trigger"),
)

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

# Open device
camera = open_device(dev)

capability_info = camera.capability_info
capability = capability_info.capability_report

if capability is None:
    raise RuntimeError("Capability reporting is not supported on this device: ")

support_features = [
    label
    for feature, label in CAPABILITY_LABELS
    if capability.capability_bits & feature
]

print(
    "\ndevice capabilities:\n"
    f"  source = {capability_info.source.name}\n"
    f"  protocol version = {capability.major_version}.{capability.minor_version}\n"
    f"  capability bits = 0x{int(capability.capability_bits):016X}\n"
    f"  supported features = {', '.join(support_features) if support_features else 'None'}"
)

flash = capability.flash
if flash is None:
    print("\nflash capability: not supported")
else:
    print(
        "\nflash capability:\n"
        f"  user space start = 0x{flash.user_space_start:08X}\n"
        f"  user space length = {flash.user_space_length} bytes"
    )

imu = capability.imu
if imu is None:
    print("\nIMU capability: not supported")
else:
    conversion = imu.conversion
    print(
        "\nIMU capability:\n"
        f"  model = {imu.model or '(not reported)'}\n"
        "  XU:\n"
        f"    unit = {imu.xu_unit}\n"
        f"    selector = {imu.xu_selector}\n"
        f"    length = {imu.xu_length}\n"
        f"    decimal exponent = {imu.decimal_exponent}\n"
        "  conversion:\n"
        f"    temperature offset = {conversion.temperature_offset_c} °C\n"
        f"    temperature LSB per °C = {conversion.temperature_lsb_per_c}\n"
        f"    accelerometer LSB per g = {conversion.accel_lsb_per_g}\n"
        f"    gyroscope LSB per °/s = {conversion.gyro_lsb_per_dps}"
    )
