# stereo_match Setup

Real-time stereo disparity and ranging for Arducam UVC Stereo cameras.

Loads calibration from camera flash, runs rectification and stereo matching on the live stream, and shows left/right or disparity in a GUI. Interactive controls adjust matching parameters, filters, confidence, and hole filling.

Optional YOLO (ultralytics) highlights detections and estimates X/Y/Z from the current disparity.


> **Platform Requirement:** This demo runs YOLO inference via `ultralytics`. It can only run on platforms that support YOLO inference (e.g., x86_64 PC or other platforms with compatible Python/PyTorch wheels).

## Environment Setup

Recommend using [uv](https://docs.astral.sh/uv/) to manage the environment.

## Run Stereo Match Demo

```bash
uv sync
uv run demo.py
```

