#include "arducam_uvc_stereo.hpp"

#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <utility>

namespace {

namespace stereo = arducam::uvc_stereo;

using CapabilityLabel = std::pair<stereo::ReportedDeviceCapability, std::string_view>;

constexpr std::array<CapabilityLabel, 5> capability_labels{{
    {stereo::ReportedDeviceCapability::kVideo, "Video"},
    {stereo::ReportedDeviceCapability::kFlash, "Flash"},
    {stereo::ReportedDeviceCapability::kImu, "IMU"},
    {stereo::ReportedDeviceCapability::kMicrophone, "Microphone"},
    {stereo::ReportedDeviceCapability::kExtTrigger, "External trigger"},
}};

const char *capability_source_name(const stereo::CapabilitySource source) noexcept
{
    switch (source) {
    case stereo::CapabilitySource::kUnknown:
        return "UNKNOWN";
    case stereo::CapabilitySource::kXuReport:
        return "XU_REPORT";
    case stereo::CapabilitySource::kFallback:
        return "FALLBACK";
    }
    return "UNKNOWN";
}

bool has_capability(const std::uint64_t capability_bits,
                    const stereo::ReportedDeviceCapability capability) noexcept
{
    return (capability_bits & static_cast<std::uint64_t>(capability)) != 0U;
}

void print_opencv_backends(const stereo::DeviceInfo &device)
{
    std::cout << '[';
    bool first = true;
    for (const auto &[backend, index] : device.opencv_backend_indices) {
        if (!first) {
            std::cout << ", ";
        }
        std::cout << "{'" << stereo::opencv_backend_name(backend) << "': " << index << '}';
        first = false;
    }
    std::cout << ']';
}

void print_device(const std::size_t index, const stereo::DeviceInfo &device)
{
    std::cout << "device[" << index << "]: vid=0x" << std::hex << std::setw(4)
              << std::setfill('0') << device.vid << " pid=0x" << std::setw(4) << device.pid
              << std::dec << " node=" << device.video_node
              << " bus=" << static_cast<unsigned int>(device.bus_number)
              << " address=" << static_cast<unsigned int>(device.device_address)
              << ", opencv=";
    print_opencv_backends(device);
    std::cout << '\n';
}

void print_selected_device(const stereo::DeviceInfo &device)
{
    std::cout << "selected device: vid=0x" << std::hex << std::setw(4)
              << std::setfill('0') << device.vid << " pid=0x" << std::setw(4) << device.pid
              << std::dec << " node=" << device.video_node
              << " bus=" << static_cast<unsigned int>(device.bus_number)
              << " address=" << static_cast<unsigned int>(device.device_address) << '\n';
}

void print_supported_features(const std::uint64_t capability_bits)
{
    bool found = false;
    for (const auto &[capability, label] : capability_labels) {
        if (!has_capability(capability_bits, capability)) {
            continue;
        }
        if (found) {
            std::cout << ", ";
        }
        std::cout << label;
        found = true;
    }
    if (!found) {
        std::cout << "None";
    }
}

void print_capability_report(const stereo::DeviceCapabilityInfo &capability_info,
                             const stereo::CapabilityReport &capability)
{
    std::cout << "\ndevice capabilities:\n"
              << "  source = " << capability_source_name(capability_info.source) << '\n'
              << "  protocol version = " << static_cast<unsigned int>(capability.major_version)
              << '.' << static_cast<unsigned int>(capability.minor_version) << '\n'
              << "  capability bits = 0x" << std::hex << std::uppercase << std::setw(16)
              << std::setfill('0') << capability.capability_bits << std::dec << std::nouppercase
              << "\n  supported features = ";
    print_supported_features(capability.capability_bits);
    std::cout << '\n';

    if (!capability.flash.has_value()) {
        std::cout << "\nflash capability: not supported\n";
    } else {
        const stereo::ReportedFlashInfo &flash = capability.flash.value();
        std::cout << "\nflash capability:\n"
                  << "  user space start = 0x" << std::hex << std::uppercase << std::setw(8)
                  << std::setfill('0') << flash.user_space_start << std::dec << std::nouppercase
                  << "\n  user space length = " << flash.user_space_length << " bytes\n";
    }

    if (!capability.imu.has_value()) {
        std::cout << "\nIMU capability: not supported\n";
        return;
    }

    const stereo::ReportedImuInfo &imu = capability.imu.value();
    const stereo::ImuConversionOptions &conversion = imu.conversion;
    std::cout << "\nIMU capability:\n"
              << "  model = " << (imu.model.empty() ? "(not reported)" : imu.model) << '\n'
              << "  XU:\n"
              << "    unit = " << static_cast<unsigned int>(imu.xu_unit) << '\n'
              << "    selector = " << static_cast<unsigned int>(imu.xu_selector) << '\n'
              << "    length = " << static_cast<unsigned int>(imu.xu_length) << '\n'
              << "    decimal exponent = " << static_cast<unsigned int>(imu.decimal_exponent)
              << "\n  conversion:\n"
              << "    temperature offset = " << conversion.temperature_offset_c << " °C\n"
              << "    temperature LSB per °C = " << conversion.temperature_lsb_per_c << '\n'
              << "    accelerometer LSB per g = " << conversion.accel_lsb_per_g << '\n'
              << "    gyroscope LSB per °/s = " << conversion.gyro_lsb_per_dps << '\n';
}

} // namespace

int main()
{
    auto scanned = stereo::scan_devices();
    if (!scanned.ok()) {
        std::cerr << "scan failed: " << scanned.status().message() << '\n';
        return 1;
    }

    const auto &devices = scanned.value();
    if (devices.empty()) {
        std::cerr << "no devices found\n";
        return 2;
    }

    for (std::size_t index = 0; index < devices.size(); ++index) {
        print_device(index, devices[index]);
    }

    const stereo::DeviceInfo &selected_device = devices.front();
    print_selected_device(selected_device);

    auto opened = stereo::open_device(selected_device);
    if (!opened.ok()) {
        std::cerr << "open_device failed: " << opened.status().message() << '\n';
        return 3;
    }

    stereo::Device device = opened.move_value();
    const stereo::DeviceCapabilityInfo &capability_info = device.capability_info();
    if (!capability_info.capability_report.has_value()) {
        std::cerr << (capability_info.message.empty()
                          ? "Capability reporting is not supported on this device"
                          : capability_info.message)
                  << '\n';
        return 4;
    }

    print_capability_report(capability_info, capability_info.capability_report.value());
    return 0;
}
