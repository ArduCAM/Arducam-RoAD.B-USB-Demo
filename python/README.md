# Python Demo

## Installation dependence

```bash
cd python
python -m pip install -r requirements.txt
```

## Demo

### read_calib_data.py

Scan for connected devices and read the calibration JSON stored on the first device found.

```bash
python read_calib_data.py
```

Expected output:

```
device[0]: vid=0x... pid=0x... node=/dev/video0 bus=1 address=2
version=0
json={...}
```

### write_calib_data.py

Write a calibration JSON file to the selected device found, then read it back to verify.

The demo reads `../calib_example.json` by default. You can replace it with your own calibration file.

```bash
python write_calib_data.py
```

Expected output:

```
selected device: vid=0x... pid=0x... node=/dev/video0 bus=1 address=2
write_json success
read version=0
read json={...}
```
