#include "arduacm_uvc_stereo.h"
#include <stdio.h>
#include <string.h>

int main(void)
{
    /* Scan for devices */
    arducam_uvc_stereo_device_info_t *devices = NULL;
    size_t count = 0;
    int32_t rc = arducam_uvc_stereo_scan_devices(&devices, &count);
    if (rc != ARDUCAM_UVC_STEREO_OK) {
        const char *msg = "";
        size_t msg_len = 0;
        arducam_uvc_stereo_last_error_message(&msg, &msg_len);
        fprintf(stderr, "scan failed: %.*s\n", (int)msg_len, msg);
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

    /* Open the selected device*/
    arducam_uvc_stereo_device_t *opened = NULL;
    rc = arducam_uvc_stereo_open_device(&select_device, NULL, &opened);
    if (rc != ARDUCAM_UVC_STEREO_OK) {
        const char *msg = "";
        size_t msg_len = 0;
        arducam_uvc_stereo_last_error_message(&msg, &msg_len);
        fprintf(stderr, "open_device failed: %.*s\n", (int)msg_len, msg);
        return 3;
    }

    /* Read calibration data from the device */
    uint16_t version = 0;
    char *json = NULL;
    size_t json_len = 0;
    rc = arducam_uvc_stereo_device_read_json(opened, &version, &json, &json_len);
    if (rc != ARDUCAM_UVC_STEREO_OK) {
        const char *msg = "";
        size_t msg_len = 0;
        arducam_uvc_stereo_last_error_message(&msg, &msg_len);
        fprintf(stderr, "read failed: %.*s\n", (int)msg_len, msg);
        arducam_uvc_stereo_close_device(opened);
        return 3;
    }

    printf("Calibration version: %u\n", (unsigned)version);
    printf("Calibration JSON:\n");
    printf("%.*s\n", (int)json_len, json);
    arducam_uvc_stereo_free(json);
    arducam_uvc_stereo_close_device(opened);
    return 0;
}