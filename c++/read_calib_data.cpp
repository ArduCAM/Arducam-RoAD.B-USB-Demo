#include "arducam_uvc_stereo.hpp"
#include <iomanip>
#include <iostream>

int main()
{
    using namespace arducam::uvc_stereo;

    UVCStereo sdk;

    /* Scan for devices */
    Result<std::vector<DeviceInfo>> scanned = sdk.scan();
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
        std::cout << "device[" << i << "]: vid=0x" << std::hex << std::setw(4)
                  << std::setfill('0') << devices[i].vid << " pid=0x" << std::setw(4)
                  << devices[i].pid << std::dec << " node=" << devices[i].video_node
                  << " bus=" << (unsigned)devices[i].bus_number
                  << " address=" << (unsigned)devices[i].device_address << "\n";
    }

    /* Select the first device */
    const DeviceInfo &dev = devices.front();

    /* Read calibration data from the device */
    Result<ReadJsonResult> r = sdk.read_json(dev);
    if (!r.ok()) {
        std::cerr << "read failed: " << r.status().message() << "\n";
        return 3;
    }

    std::cout << "version=" << r.value().version << "\n";
    std::cout << "json=" << r.value().json_utf8 << "\n";
    return 0;
}