# QXG Platform

QXG Platform turns video detections into qualitative explainable graphs. It combines object detection/tracking, optional depth input, qualitative spatial relations, relevance filtering, and an OpenCV visualization loop.

This repository is a production-oriented copy of the original research workspace. It intentionally excludes model weights, recordings, generated media, huge datasets, compiled binaries, and presentation files.

## What Is Included

- `src/qxg_platform/`: maintainable Python package
- `configs/`: environment-specific YAML configs
- `tests/`: automated tests for core graph behavior and config loading
- `docs/`: operational notes for model artifacts and deployment

## What Is Not Included

Large artifacts should live outside Git, or in Git LFS/DVC:

- YOLO weights: `*.pt`, `*.pth`
- sklearn/joblib models
- RealSense recordings
- generated videos, GIFs, PDFs, PPTX files
- training datasets

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

Install ML extras only on machines that run inference:

```powershell
python -m pip install -e ".[ml]"
```

For RealSense camera support:

```powershell
python -m pip install -e ".[realsense]"
```

## Run

Graphical launcher:

```powershell
qxg-gui
```

The launcher lets you choose `video`, `camera`, `realsense`, or `recording`, pick or override the model profile, and then start the visualization dashboard.

Recorded/video-directory mode:

```powershell
qxg --config configs/video.yaml --input recording --source D:\path\to\recording
```

Single video file:

```powershell
qxg --config configs/video.yaml --input video --source D:\path\to\video.mp4
```

Normal webcam:

```powershell
qxg --config configs/video.yaml --input camera --source 0
```

Server mode:

```powershell
qxg-server --config configs/realtime.yaml --host 127.0.0.1 --port 5000
```

Client mode:

```powershell
qxg --mode remote --config configs/realtime.yaml --server-url http://127.0.0.1:5000
```

## Engineering Notes

- Network payloads use JSON plus base64 arrays, not pickle.
- Model files are configured by path and validated at startup.
- GUI model profiles live in `configs/model_profiles.yaml`.
- The core QXG relation engine has a pure-Python implementation with tests.
- Optional heavy dependencies are loaded lazily so tests and packaging work on clean machines.
