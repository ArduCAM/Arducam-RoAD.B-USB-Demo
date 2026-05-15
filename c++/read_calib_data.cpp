#include "arducam_uvc_stereo.hpp"
#include <iomanip>
#include <iostream>

int main()
{
    using namespace arducam::uvc_stereo;

    /* Scan for devices */
    Result<std::vector<DeviceInfo>> scanned = scan_devices();
    if (!scanned.ok()) {
        std::cerr << "scan failed: " << scanned.status().message() << "\n";
        return 1;
    }
    if (scanned.value().empty()) {
        std::cerr << "no devices found\n";
        return 2;
    }

    /* Print device list */
    const std::vector<DeviceInfo> &devices = scanned.value();
    for (size_t i = 0; i < devices.size(); ++i) {
        const DeviceInfo &device = devices[i];
        std::cout << "device[" << i << "]: vid=0x" << std::hex << std::setw(4)
                  << std::setfill('0') << device.vid << " pid=0x" << std::setw(4)
                  << device.pid << std::dec << " node=" << device.video_node
                  << " bus=" << (unsigned)device.bus_number
                  << " address=" << (unsigned)device.device_address << "\n";
    }

    /* Select and open the first device */
    const DeviceInfo &select_device = devices.front();
    std::cout << "selected device: vid=0x" << std::hex << std::setw(4)
              << std::setfill('0') << select_device.vid << " pid=0x" << std::setw(4)
              << select_device.pid << std::dec << " node=" << select_device.video_node
              << " bus=" << (unsigned)select_device.bus_number
              << " address=" << (unsigned)select_device.device_address << "\n";

    /* Read calibration data from the device */
    Result<Device> opened = open_device(select_device);
    if (!opened.ok()) {
        std::cerr << "open_device failed: " << opened.status().message() << "\n";
        return 3;
    }
    Device device = opened.move_value();

    Result<ReadJsonResult> r = device.read_json();
    if (!r.ok()) {
        std::cerr << "read failed: " << r.status().message() << "\n";
        return 3;
    }

    std::cout << "Calibration version: " << r.value().version << "\n";
    std::cout << "Calibration JSON:\n";
    std::cout << r.value().json_utf8 << "\n";
    return 0;
}
