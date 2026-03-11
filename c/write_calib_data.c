#include "arduacm_uvc_stereo.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_last_error(const char *tag, int32_t code)
{
    const char *msg = "";
    size_t msg_len = 0;
    (void)arducam_uvc_stereo_last_error_message(&msg, &msg_len);
    fprintf(stderr, "%s failed: code=%d, message=%.*s\n",
            tag, (int)code, (int)msg_len, msg == NULL ? "" : msg);
}

int main(void)
{
    /* Scan for devices */
    arducam_uvc_stereo_device_info_t *devices = NULL;
    size_t count = 0;
    int32_t rc = arducam_uvc_stereo_scan_devices(&devices, &count);
    if (rc != ARDUCAM_UVC_STEREO_OK) {
        print_last_error("arducam_uvc_stereo_scan_devices", rc);
        return 1;
    }
    if (count == 0) {
        fprintf(stderr, "no devices found\n");
        arducam_uvc_stereo_free(devices);
        return 2;
    }

    arducam_uvc_stereo_device_info_t dev = devices[0];
    arducam_uvc_stereo_free(devices);

    printf("selected device: vid=0x%04x pid=0x%04x node=%s bus=%u address=%u\n",
           (unsigned)dev.vid, (unsigned)dev.pid, dev.video_node,
           (unsigned)dev.bus_number, (unsigned)dev.device_address);

    /* Load calibration JSON from a local file */
    FILE *fin = fopen("../../calib_example.json", "rb");
    if (!fin) {
        fprintf(stderr, "failed to open calib_example.json\n");
        return 3;
    }

    fseek(fin, 0, SEEK_END);
    long n = ftell(fin);
    fseek(fin, 0, SEEK_SET);

    char *json = (char *)malloc((size_t)n + 1);
    if (!json) {
        fclose(fin);
        fprintf(stderr, "malloc failed\n");
        return 4;
    }

    fread(json, 1, (size_t)n, fin);
    fclose(fin);
    json[n] = '\0';

    /* Write calibration data to the device */
    rc = arducam_uvc_stereo_write_json(&dev, json, (size_t)n);
    free(json);
    if (rc != ARDUCAM_UVC_STEREO_OK) {
        print_last_error("arducam_uvc_stereo_write_json", rc);
        return 5;
    }
    printf("write_json success\n");

    /* Read back to verify */
    uint16_t version = 0;
    char *json_out = NULL;
    size_t json_out_len = 0;
    rc = arducam_uvc_stereo_read_json(&dev, &version, &json_out, &json_out_len);
    if (rc != ARDUCAM_UVC_STEREO_OK) {
        print_last_error("arducam_uvc_stereo_read_json", rc);
        return 6;
    }

    printf("read version=%u\n", (unsigned)version);
    printf("read json=%.*s\n", (int)json_out_len, json_out);
    arducam_uvc_stereo_free(json_out);
    return 0;
}