# stereo_match Setup

Real-time stereo disparity and ranging for Arducam UVC Stereo cameras.

Loads calibration from camera flash, runs rectification and stereo matching on the live stream, and shows left/right or disparity in a GUI. Interactive controls adjust matching parameters, filters, confidence, and hole filling.

## Installation dependence

```bash
python -m pip install -r requirements.txt
```

## Run
```bash
python demo.py
```