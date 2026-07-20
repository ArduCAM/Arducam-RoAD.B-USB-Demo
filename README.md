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

```
.
├── c/                                   # C demos
│   ├── CMakeLists.txt
│   ├── build.sh
│   ├── read_calib_data.c                # Read calibration data
│   ├── write_calib_data.c               # Write calibration data
│   ├── read_capability.c                # Read device capabilities
│   └── read_imu.c                       # Read IMU data
│
├── c++/                                 # C++ demos
│   ├── CMakeLists.txt
│   ├── build.sh
│   ├── read_calib_data.cpp              # Read calibration data
│   ├── write_calib_data.cpp             # Write calibration data
│   ├── read_capability.cpp              # Read device capabilities
│   └── read_imu.cpp                     # Read IMU data
│
├── python/                              # Python demos
│   ├── calibration/                     # Mono and stereo calibration tools
│   │   ├── mono_calib.py
│   │   └── stereo_calib.py
│   ├── imu/
│   │   └── read_imu.py                  # Read IMU data
│   ├── undistort                        # Image rectification demo
│   ├── read_calib_data.py               # Read calibration data
│   ├── write_calib_data.py              # Write calibration data
│   └── read_capability.py               # Read device capabilities
│
├── advanced_demo/                       # Advanced demos
│   ├── stereo_matc                      # Stereo matching demo
│   ├── imu_axis3d_viewer                # IMU 3D-axis viewer
│   └── stereo_imu_pose_viewer           # Stereo-IMU pose viewer
│
└── ros2/                                # ROS 2 Stereo Demo 
```

## Basic Demos:
- [C Demo](c/README.md)
- [C++ Demo](c++/README.md)
- [Python Demo](python/README.md)

## Advanced Demos:
- [Stereo Match Demo](advanced_demo/stereo_match/README.md)
- [IMU 3D Axis Viewer Demo](advanced_demo/imu_axis3d_viewer/README.md)
- [Stereo IMU Pose Viewer Demo](advanced_demo/stereo_imu_pose_viewer/README.md)
- [ROS2 Demo](ros2/README.md)
