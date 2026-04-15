# C Demo

## Installation dependence

### C

#### Debian 13 (trixie) and above versions

```
sudo curl -s --compressed -o /usr/share/keyrings/arducam-keyring.pgp "https://arducam.github.io/arducam_ppa/KEY.gpg"
sudo curl -s --compressed -o /etc/apt/sources.list.d/arducam_list_files.sources "https://arducam.github.io/arducam_ppa/arducam_list_files.sources"
sudo apt update
sudo apt install arducam-config-parser-dev arducam-usb-sdk-dev
```

#### Other

```
curl -s --compressed "https://arducam.github.io/arducam_ppa/KEY.gpg" | sudo apt-key add -
sudo curl -s --compressed -o /etc/apt/sources.list.d/arducam_list_files.list "https://arducam.github.io/arducam_ppa/arducam_list_files.list"
sudo apt update
sudo apt install arducam-config-parser-dev arducam-usb-sdk-dev

## Build

```bash
cd c
./build.sh
```

The binaries are output to `build/`.

## Demo

### read_calib_data

Scan for connected devices and read the calibration JSON stored on the first device found.

```bash
./build/read_calib_data
```

Expected output:

```
device[0]: vid=0x... pid=0x... node=/dev/video0 bus=1 address=2
version=0
json={...}
```

### write_calib_data

Write a calibration JSON file to the selected device found, then read it back to verify.

The demo reads `../../calib_example.json` by default. You can replace it with your own calibration file.

```bash
./build/write_calib_data
```

Expected output:

```
selected device: vid=0x... pid=0x... node=/dev/video0 bus=1 address=2
write_json success
read version=0
read json={...}
```
