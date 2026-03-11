#include "arducam_uvc_stereo.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

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

    const DeviceInfo &dev = scanned.value().front();
    std::cout << "selected device: vid=0x" << std::hex << std::setw(4)
              << std::setfill('0') << dev.vid << " pid=0x" << std::setw(4)
              << dev.pid << std::dec << " node=" << dev.video_node
              << " bus=" << (unsigned)dev.bus_number
              << " address=" << (unsigned)dev.device_address << "\n";

    /* Load calibration JSON from a local file */
    std::ifstream fin("../../calib_example.json");
    if (!fin) {
        std::cerr << "failed to open calib_example\n";
        return 3;
    }

    std::ostringstream oss;
    oss << fin.rdbuf();
    std::string content = oss.str();

    /* Write calibration data to the device */
    Status st = sdk.write_json(content, dev);
    if (!st.ok()) {
        std::cerr << "write_json failed: " << st.message() << "\n";
        return 4;
    }
    std::cout << "write_json success\n";

    /* Read back to verify */
    Result<ReadJsonResult> r = sdk.read_json(dev);
    if (!r.ok()) {
        std::cerr << "read_json failed: " << r.status().message() << "\n";
        return 5;
    }

    std::cout << "read version=" << r.value().version << "\n";
    std::cout << "read json=" << r.value().json_utf8 << "\n";
    return 0;
}
