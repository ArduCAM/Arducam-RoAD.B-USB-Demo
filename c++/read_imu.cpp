
#include "arducam_uvc_stereo.hpp"
#include <iomanip>
#include <iostream>

int main()
{
    using namespace arducam::uvc_stereo;

    Result<std::vector<DeviceInfo>> scanned = scan_devices();
    if (!scanned.ok()) {
        std::cerr << "scan failed: " << scanned.status().message() << "\n";
        return 1;
    }
    if (scanned.value().empty()) {
        std::cerr << "no devices found\n";
        return 2;
    }

    const std::vector<DeviceInfo> &devices = scanned.value();
    for (size_t index = 0; index < devices.size(); ++index) {
        const DeviceInfo &device = devices[index];
        std::cout << "device[" << index << "]: vid=0x" << std::hex << std::setw(4)
                  << std::setfill('0') << device.vid << " pid=0x" << std::setw(4)
                  << device.pid << std::dec << " node=" << device.video_node
                  << " bus=" << (unsigned)device.bus_number
                  << " address=" << (unsigned)device.device_address << "\n";
    }

    const DeviceInfo &select_device = devices.front();
    std::cout << "selected device: vid=0x" << std::hex << std::setw(4)
              << std::setfill('0') << select_device.vid << " pid=0x" << std::setw(4)
              << select_device.pid << std::dec << " node=" << select_device.video_node
              << " bus=" << (unsigned)select_device.bus_number
              << " address=" << (unsigned)select_device.device_address << "\n";

    Result<Device> opened = open_device(select_device);
    if (!opened.ok()) {
        std::cerr << "open_device failed: " << opened.status().message() << "\n";
        return 3;
    }
    Device device = opened.move_value();
    Status imu_status = device.open_imu();
    if (!imu_status.ok()) {
        std::cerr << "open_imu failed: " << imu_status.message() << "\n";
        return 3;
    }

    Result<ReadImuResult> raw_result = device.read_imu();
    if (!raw_result.ok()) {
        std::cerr << "read_imu failed: " << raw_result.status().message() << "\n";
        device.close_imu();
        return 4;
    }
    const ReadImuResult &raw = raw_result.value();

    Result<ConvertedImuResult> converted_result = convert_imu(raw);
    if (!converted_result.ok()) {
        std::cerr << "convert_imu failed: " << converted_result.status().message() << "\n";
        device.close_imu();
        return 5;
    }
    const ConvertedImuResult &converted = converted_result.value();

    std::cout << "IMU data:\n"
              << "  temperature_raw: " << raw.temperature_raw << "\n"
              << "  temperature_c:   " << std::fixed << std::setprecision(3)
              << converted.temperature_c << "\n"
              << "  accel_raw:       x=" << raw.accel_x_raw
              << " y=" << raw.accel_y_raw
              << " z=" << raw.accel_z_raw << "\n"
              << "  gyro_raw:        x=" << raw.gyro_x_raw
              << " y=" << raw.gyro_y_raw
              << " z=" << raw.gyro_z_raw << "\n";

    device.close_imu();
    return 0;
}
