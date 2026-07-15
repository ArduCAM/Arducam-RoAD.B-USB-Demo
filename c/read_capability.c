#include "arduacm_uvc_stereo.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct capability_label {
    uint64_t capability;
    const char *label;
} capability_label_t;

static const capability_label_t capability_labels[] = {
    {ARDUCAM_UVC_STEREO_REPORTED_CAPABILITY_VIDEO, "Video"},
    {ARDUCAM_UVC_STEREO_REPORTED_CAPABILITY_FLASH, "Flash"},
    {ARDUCAM_UVC_STEREO_REPORTED_CAPABILITY_IMU, "IMU"},
    {ARDUCAM_UVC_STEREO_REPORTED_CAPABILITY_MICROPHONE, "Microphone"},
    {ARDUCAM_UVC_STEREO_REPORTED_CAPABILITY_EXT_TRIGGER, "External trigger"},
};

static void print_last_error(const char *operation)
{
    const char *message = "";
    size_t message_length = 0U;

    (void)arducam_uvc_stereo_last_error_message(&message, &message_length);
    fprintf(stderr,
            "%s failed: %.*s\n",
            operation,
            (int)message_length,
            message == NULL ? "" : message);
}

static const char *capability_source_name(arducam_uvc_stereo_capability_source_t source)
{
    switch (source) {
    case ARDUCAM_UVC_STEREO_CAPABILITY_SOURCE_UNKNOWN:
        return "UNKNOWN";
    case ARDUCAM_UVC_STEREO_CAPABILITY_SOURCE_XU_REPORT:
        return "XU_REPORT";
    case ARDUCAM_UVC_STEREO_CAPABILITY_SOURCE_FALLBACK:
        return "FALLBACK";
    default:
        return "UNKNOWN";
    }
}

static void print_opencv_backends(const arducam_uvc_stereo_device_info_t *device)
{
    size_t index;

    putchar('[');
    for (index = 0U; index < device->opencv_backend_index_count; ++index) {
        const arducam_uvc_stereo_opencv_backend_index_entry_t *entry =
            &device->opencv_backend_indices[index];
        const char *name = arducam_uvc_stereo_opencv_backend_name(entry->backend);
        printf("%s{'%s': %d}", index == 0U ? "" : ", ", name, (int)entry->index);
    }
    putchar(']');
}

static void print_device(size_t index, const arducam_uvc_stereo_device_info_t *device)
{
    printf("device[%zu]: vid=0x%04x pid=0x%04x node=%s bus=%u address=%u, opencv=",
           index,
           (unsigned int)device->vid,
           (unsigned int)device->pid,
           device->video_node,
           (unsigned int)device->bus_number,
           (unsigned int)device->device_address);
    print_opencv_backends(device);
    putchar('\n');
}

static void print_selected_device(const arducam_uvc_stereo_device_info_t *device)
{
    printf("selected device: vid=0x%04x pid=0x%04x node=%s bus=%u address=%u\n",
           (unsigned int)device->vid,
           (unsigned int)device->pid,
           device->video_node,
           (unsigned int)device->bus_number,
           (unsigned int)device->device_address);
}

static void print_supported_features(uint64_t capability_bits)
{
    size_t index;
    int found = 0;

    for (index = 0U; index < sizeof(capability_labels) / sizeof(capability_labels[0]); ++index) {
        if ((capability_bits & capability_labels[index].capability) == 0U) {
            continue;
        }
        printf("%s%s", found != 0 ? ", " : "", capability_labels[index].label);
        found = 1;
    }
    if (found == 0) {
        fputs("None", stdout);
    }
}

static void print_capability_report(const arducam_uvc_stereo_capability_info_t *info)
{
    printf("\ndevice capabilities:\n");
    printf("  source = %s\n", capability_source_name(info->source));
    printf("  protocol version = %u.%u\n",
           (unsigned int)info->major_version,
           (unsigned int)info->minor_version);
    printf("  capability bits = 0x%016" PRIX64 "\n", info->capability_bits);
    fputs("  supported features = ", stdout);
    print_supported_features(info->capability_bits);
    putchar('\n');

    if (info->has_flash_info == 0U) {
        printf("\nflash capability: not supported\n");
    } else {
        printf("\nflash capability:\n");
        printf("  user space start = 0x%08" PRIX32 "\n", info->flash_user_space_start);
        printf("  user space length = %" PRIu32 " bytes\n", info->flash_user_space_length);
    }

    if (info->has_imu_info == 0U) {
        printf("\nIMU capability: not supported\n");
        return;
    }

    printf("\nIMU capability:\n");
    printf("  model = %s\n", info->imu_model[0] == '\0' ? "(not reported)" : info->imu_model);
    printf("  XU:\n");
    printf("    unit = %u\n", (unsigned int)info->imu_xu_unit);
    printf("    selector = %u\n", (unsigned int)info->imu_xu_selector);
    printf("    length = %u\n", (unsigned int)info->imu_xu_length);
    printf("    decimal exponent = %u\n", (unsigned int)info->imu_decimal_exponent);
    printf("  conversion:\n");
    printf("    temperature offset = %.15g °C\n", info->temperature_offset_c);
    printf("    temperature LSB per °C = %.15g\n", info->temperature_lsb_per_c);
    printf("    accelerometer LSB per g = %.15g\n", info->accel_lsb_per_g);
    printf("    gyroscope LSB per °/s = %.15g\n", info->gyro_lsb_per_dps);
}

int main(void)
{
    arducam_uvc_stereo_device_info_t *devices = NULL;
    arducam_uvc_stereo_device_info_t selected_device;
    arducam_uvc_stereo_device_t *opened = NULL;
    arducam_uvc_stereo_capability_info_t capability_info;
    size_t count = 0U;
    size_t index;
    int32_t result;

    result = arducam_uvc_stereo_scan_devices(&devices, &count);
    if (result != ARDUCAM_UVC_STEREO_OK) {
        print_last_error("scan");
        return 1;
    }
    if (count == 0U) {
        fprintf(stderr, "no devices found\n");
        arducam_uvc_stereo_free(devices);
        return 2;
    }

    for (index = 0U; index < count; ++index) {
        print_device(index, &devices[index]);
    }

    selected_device = devices[0];
    arducam_uvc_stereo_free(devices);
    print_selected_device(&selected_device);

    result = arducam_uvc_stereo_open_device(&selected_device, NULL, &opened);
    if (result != ARDUCAM_UVC_STEREO_OK) {
        print_last_error("open_device");
        return 3;
    }

    arducam_uvc_stereo_capability_info_init(&capability_info);
    result = arducam_uvc_stereo_device_capability_info(opened, &capability_info);
    if (result != ARDUCAM_UVC_STEREO_OK) {
        print_last_error("capability_info");
        arducam_uvc_stereo_close_device(opened);
        return 4;
    }
    if (capability_info.has_capability_report == 0U) {
        fprintf(stderr,
                "%s\n",
                capability_info.message[0] == '\0'
                    ? "Capability reporting is not supported on this device"
                    : capability_info.message);
        arducam_uvc_stereo_close_device(opened);
        return 4;
    }

    print_capability_report(&capability_info);
    arducam_uvc_stereo_close_device(opened);
    return 0;
}
