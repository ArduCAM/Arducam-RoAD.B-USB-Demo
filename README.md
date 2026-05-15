# Arducam RoAD.B-USB Demo

Demo code for reading and writing calibration data on Arducam UVC stereo cameras, available in C, C++, and Python.


## API documentation

For detailed API reference and usage instructions, please refer to the [Arducam online documentation](https://www.arducam.com/docs/Arducam-RoAD.B-USB).

## Installation dependence

### Linux
Please refer to the [Linux installation](doc/linux_environmental_install.md).

## Quick Start
To run the demo in the fastest and simplest way, it is recommended to use the [Python demo](./python/README.md) (please set [udev rules](./doc/linux_environmental_install.md###set-udev-rules) and [Python dependencies](./python/README.md###installation-dependence) before trying it).

## Notes
- After reading or writing the calibration file, starting the camera stream may fail. Please add a short delay before starting streaming again.



## What's next
The project contains multiple demos, please try them according to the corresponding descriptions.

> **Note:** These demos require Arducam UVC Stereo SDK version **> 0.2**.

```.
├── c/                          # C Demo
│   ├── build.sh                # Build script
│   ├── CMakeLists.txt
│   ├── read_calib_data.c       # Read calibration data from device
│   ├── write_calib_data.c      # Write calibration data to device
│   ├── read_imu.c              # Read IMU data from device
│   └── README.md
├── c++/                        # C++ Demo
│   ├── build.sh                # Build script
│   ├── CMakeLists.txt
│   ├── read_calib_data.cpp     # Read calibration data from device
│   ├── write_calib_data.cpp    # Write calibration data to device
│   ├── read_imu.cpp            # Read IMU data from device
│   └── README.md
├── python/                     # Python Demo
│   ├── requirements.txt        # Python dependencies
│   ├── read_calib_data.py      # Read calibration data from device
│   ├── write_calib_data.py     # Write calibration data to device
│   ├── calibration/            # Stereo calibration tool
│   ├── undistort/              # Rectification demo
│   ├── imu/                    # IMU demo
│   │   └── read_imu.py         # Read IMU data from device
│   └── README.md
├── advanced_demo/              # Advanced Demos
│   ├── stereo_match/           # Stereo Match Demo
│   ├── imu_axis3d_viewer/      # IMU 3D Axis Viewer Demo
│   │   ├── demo.py
│   │   ├── requirements.txt
│   │   └── utils/
│   └── README.md
├── ros2/                       # ROS2 Demo
└── calib_example.json          # Example calibration data file
```

## Basic Demos:
- [C Demo](c/README.md)
- [C++ Demo](c++/README.md)
- [Python Demo](python/README.md)

## Advanced Demos:
- [Stereo Match Demo](advanced/stereo_match/README.md)
- [IMU 3D Axis Viewer Demo](advanced/imu_axis3d_viewer/README.md)
- [ROS2 Demo](ros2/README.md)
