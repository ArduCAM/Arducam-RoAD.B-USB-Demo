# stereo_match Setup

`stereo_match` depends on the OpenCV contrib `ximgproc` module. Create a dedicated virtual environment for this demo instead of sharing the environment with other Python examples in this repository.

## Environment Setup

Run the following commands from the repository root:

```bash
cd python/stereo_match
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip uninstall -y opencv-python
python -m pip install opencv-contrib-python==4.13.0.92
```

## Notes

- Use a separate virtual environment to avoid dependency conflicts with other examples under `python/`.
- After `pip install -r requirements.txt`, uninstall `opencv-python` if it was installed automatically by other dependencies.
- Then install `opencv-contrib-python==4.13.0.92` so the `cv2.ximgproc` features required by `stereo_match` are available.

## Run

After activating the virtual environment, run:

```bash
python demo.py
```
