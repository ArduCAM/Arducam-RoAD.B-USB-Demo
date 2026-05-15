#include "arduacm_uvc_stereo.h"
#include <stdio.h>

int main(void)
{
    /* Scan for devices */
    arducam_uvc_stereo_device_info_t *devices = NULL;
    size_t count = 0;
    int32_t rc = arducam_uvc_stereo_scan_devices(&devices, &count);
    if (rc != ARDUCAM_UVC_STEREO_OK) {
        const char *msg = "";
        size_t msg_len = 0;
        (void)arducam_uvc_stereo_last_error_message(&msg, &msg_len);
        fprintf(stderr, "scan failed: %.*s\n", (int)msg_len, msg == NULL ? "" : msg);
        return 1;
    }
    if (count == 0) {
        fprintf(stderr, "no devices found\n");
        arducam_uvc_stereo_free(devices);
        return 2;
    }

    /* Print device list */
    for (size_t i = 0; i < count; ++i) {
        printf("device[%zu]: vid=0x%04x pid=0x%04x node=%s bus=%u address=%u\n",
            i,
            (unsigned)devices[i].vid,
            (unsigned)devices[i].pid,
            devices[i].video_node,
            (unsigned)devices[i].bus_number,
            (unsigned)devices[i].device_address);
    }

    /* Select the first device */
    arducam_uvc_stereo_device_info_t select_device = devices[0];
    arducam_uvc_stereo_free(devices);

    printf("selected device: vid=0x%04x pid=0x%04x node=%s bus=%u address=%u\n",
        (unsigned)select_device.vid,
        (unsigned)select_device.pid,
        select_device.video_node,
        (unsigned)select_device.bus_number,
        (unsigned)select_device.device_address);

    /* Open selected device, then open IMU */
    arducam_uvc_stereo_device_t *opened = NULL;
    rc = arducam_uvc_stereo_open_device(&select_device, NULL, &opened);
    if (rc != ARDUCAM_UVC_STEREO_OK) {
        const char *msg = "";
        size_t msg_len = 0;
        (void)arducam_uvc_stereo_last_error_message(&msg, &msg_len);
        fprintf(stderr, "open_device failed: %.*s\n", (int)msg_len, msg == NULL ? "" : msg);
        return 3;
    }

    rc = arducam_uvc_stereo_device_open_imu(opened);
    if (rc != ARDUCAM_UVC_STEREO_OK) {
        const char *msg = "";
        size_t msg_len = 0;
        (void)arducam_uvc_stereo_last_error_message(&msg, &msg_len);
        fprintf(stderr, "open_imu failed: %.*s\n", (int)msg_len, msg == NULL ? "" : msg);
        arducam_uvc_stereo_close_device(opened);
        return 3;
    }

    /* Read IMU data */
    arducam_uvc_stereo_read_imu_result_t raw;
    rc = arducam_uvc_stereo_device_read_imu(opened, &raw);
    if (rc != ARDUCAM_UVC_STEREO_OK) {
        const char *msg = "";
        size_t msg_len = 0;
        (void)arducam_uvc_stereo_last_error_message(&msg, &msg_len);
        fprintf(stderr, "read_imu failed: %.*s\n", (int)msg_len, msg == NULL ? "" : msg);
        arducam_uvc_stereo_device_close_imu(opened);
        arducam_uvc_stereo_close_device(opened);
        return 4;
    }

    arducam_uvc_stereo_converted_imu_result_t converted;
    rc = arducam_uvc_stereo_convert_imu(&raw, NULL, &converted);
    if (rc != ARDUCAM_UVC_STEREO_OK) {
        const char *msg = "";
        size_t msg_len = 0;
        (void)arducam_uvc_stereo_last_error_message(&msg, &msg_len);
        fprintf(stderr, "convert_imu failed: %.*s\n", (int)msg_len, msg == NULL ? "" : msg);
        arducam_uvc_stereo_device_close_imu(opened);
        arducam_uvc_stereo_close_device(opened);
        return 5;
    }

    printf("IMU data:\n");
    printf("  temperature_raw: %d\n", (int)raw.temperature_raw);
    printf("  temperature_c:   %.3f\n", converted.temperature_c);
    printf("  accel_raw:       x=%d y=%d z=%d\n",
        (int)raw.accel_x_raw,
        (int)raw.accel_y_raw,
        (int)raw.accel_z_raw);
    printf("  gyro_raw:        x=%d y=%d z=%d\n",
        (int)raw.gyro_x_raw,
        (int)raw.gyro_y_raw,
        (int)raw.gyro_z_raw);

    arducam_uvc_stereo_device_close_imu(opened);
    arducam_uvc_stereo_close_device(opened);
    return 0;
}
