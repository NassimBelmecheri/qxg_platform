# QXG Platform

QXG Platform is a video perception and qualitative reasoning platform developed in the context of AI4CCAM. It detects and tracks road-scene objects, estimates or consumes depth when available, builds qualitative explainable graphs, and visualizes the resulting spatial relations in real time.

The platform is designed for connected, cooperative, and automated mobility scenarios where perception output should be interpretable, inspectable, and reusable for downstream reasoning.

## Core Capabilities

- Object detection and tracking with YOLO models.
- 2D qualitative spatial reasoning from video, webcam, or recordings.
- 3D reasoning with RealSense depth, recorded depth, or monocular depth estimation.
- Real-time OpenCV dashboard with camera view, BEV map, graph view, relation table, and object metrics.
- Relevance filtering for selecting the most important detected objects.
- Real-time recording of color frames, and depth frames when 3D depth is available.
- QXG graph export to JSON for later analysis.
- GUI object-selection panel for choosing which classes to detect.
- Local CLI mode and server/client mode for remote processing workflows.

## Repository Layout

```text
configs/              Runtime, model-profile, and tracker configuration
docs/                 Operational notes for models and deployment
models/               Local model directory, not intended for Git-tracked weights
scripts/              Utility scripts
src/qxg_platform/     Python package source
tests/                Automated tests
```

Large runtime artifacts such as model weights, datasets, recordings, generated videos, and RealSense captures should stay outside Git or be managed with Git LFS/DVC.

## Installation

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Install machine-learning dependencies on machines that run detection or monocular depth:

```powershell
python -m pip install -e ".[ml]"
```

Install RealSense support only on machines connected to an Intel RealSense camera:

```powershell
python -m pip install -e ".[realsense]"
```

To install all Python extras at once:

```powershell
python -m pip install -e ".[dev,ml,realsense]"
```

### Linux

Install system packages for Python virtual environments and OpenCV runtime support:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip libgl1 libglib2.0-0
```

Create the environment and install the platform:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install inference dependencies:

```bash
python -m pip install -e ".[ml]"
```

For RealSense input, install the Intel RealSense SDK for your Linux distribution, then install:

```bash
python -m pip install -e ".[realsense]"
```

To install all Python extras at once:

```bash
python -m pip install -e ".[dev,ml,realsense]"
```

### macOS

Install Python with Homebrew if needed:

```bash
brew install python
```

Create the environment and install the platform:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install inference dependencies:

```bash
python -m pip install -e ".[ml]"
```

RealSense support on macOS depends on your camera model, SDK support, and local driver setup. Install librealsense first if your hardware is supported, then run:

```bash
python -m pip install -e ".[realsense]"
```

To install all Python extras at once:

```bash
python -m pip install -e ".[dev,ml,realsense]"
```

### Verify Installation

Run the test suite:

```bash
pytest
```

## Model Setup

Detection model paths are configured in:

```text
configs/model_profiles.yaml
configs/video.yaml
configs/realtime.yaml
```

Update `model_weights` so each profile points to a local YOLO weights file, for example:

```yaml
model_weights: "D:/nassim/qxg_platform/models/yolo11x.pt"
```

Model weights are intentionally not stored in this repository.

## Graphical Launcher

Start the launcher:

```powershell
qxg-gui
```

The launcher lets you configure:

- Input source: `video`, `camera`, `realsense`, or `recording`.
- Source path or camera index.
- Model profile and model weights.
- Reasoning mode: `2d` or `3d`.
- Relevance filtering.
- Real-time recording output.
- QXG graph export output.
- Object classes to detect through checkboxes.

Press **Start Platform** to open the visualization dashboard. Press `q` inside the OpenCV dashboard to stop the run.

## Input Modes

### Single Video File

Use this for normal video files such as `.mp4`, `.avi`, `.mov`, or `.mkv`.

```powershell
qxg --config configs/video.yaml --input video --source D:\path\to\video.mp4
```

In 3D mode, single video files use monocular depth estimation through `depth_estimation.model_name`.

### Webcam

Use this for a standard live camera.

```powershell
qxg --config configs/video.yaml --input camera --source 0
```

`0` is the default camera index. Use `1`, `2`, and so on for additional cameras.

### RealSense

Use this for a live Intel RealSense camera.

```powershell
qxg --config configs/realtime.yaml --input realsense
```

In 3D mode, QXG uses the RealSense depth stream aligned to the color stream.

### Recording Directory

Use this to replay a saved QXG recording directory.

```powershell
qxg --config configs/video.yaml --input recording --source D:\path\to\recording
```

A recording directory uses this structure:

```text
recording/
  color/
    000001-color.jpg
    000002-color.jpg
  depth/
    000001-depth.png
    000002-depth.png
  config.json
```

For 2D recordings, only the `color/` directory is required. For 3D recordings, `depth/` and `config.json` provide depth maps and camera intrinsics. If 3D mode is selected and a recording has no depth frames, QXG automatically runs monocular depth estimation for the color frames.

## Real-Time Recording

The GUI includes a **Realtime Recording** panel. When enabled, QXG saves a session under:

```text
recordings/qxg_recording_YYYYMMDD_HHMMSS/
```

In `2d` mode, QXG saves:

```text
color/*.jpg
```

In `3d` mode, QXG saves color frames plus depth data when depth is available:

```text
color/*.jpg
depth/*.png
config.json
```

This makes live camera or RealSense runs replayable later through the `recording` input mode.

## QXG Graph Export

The GUI includes a **QXG Export** panel. When enabled, QXG saves the graph built during the run as JSON under:

```text
qxg_exports/qxg_graph_YYYYMMDD_HHMMSS.json
```

The export contains:

- `schema`: export schema name.
- `reasoning_mode`: `2d` or `3d`.
- `frames`: one entry per processed frame.
- `objects`: tracked object state for each frame.
- `relevant_object_ids`: object IDs selected by relevance filtering.
- `relations`: QXG pairwise qualitative relations, keyed as `left_id:right_id`.

Enable it from the launcher, or set it in YAML:

```yaml
qxg_export:
  enabled: true
  output_dir: "qxg_exports"
```

## Object Selection

The GUI object panel controls `detection.classes`. Only checked object classes are kept after detection. This is useful for AI4CCAM experiments that focus on specific road users such as pedestrians, cars, buses, trucks, bicycles, and motorcycles.

The default class lists live in `configs/model_profiles.yaml`. You can adjust them per model profile.

## Server And Client Mode

Start the processing server:

```powershell
qxg-server --config configs/realtime.yaml --host 127.0.0.1 --port 5000
```

Run a remote client:

```powershell
qxg --mode remote --config configs/realtime.yaml --server-url http://127.0.0.1:5000
```

Network payloads use JSON and base64-encoded arrays.

## Configuration Reference

Important sections:

- `runtime.reasoning_mode`: `2d` or `3d`.
- `depth_estimation.model_name`: monocular depth model used for non-depth cameras/videos in 3D mode.
- `realsense`: width, height, and FPS for RealSense input.
- `detection.model_weights`: YOLO model path.
- `detection.classes`: allowed object categories.
- `detection.confidence_threshold`: minimum detector confidence.
- `analysis.algebras`: qualitative relation families to compute.
- `relevance`: relevance filtering mode and thresholds.
- `visualization`: dashboard options.
- `recording`: realtime recording options.
- `qxg_export`: QXG graph JSON export options.

## References

This project builds on qualitative spatial and temporal reasoning methods and applies them to automated-driving scene interpretation in the AI4CCAM context.

### QXG And Automated Driving

- Nassim Belmecheri, Arnaud Gotlieb, Nadjib Lazaar, and Helge Spieker. [Acquiring Qualitative Explainable Graphs for Automated Driving Scene Interpretation](https://arxiv.org/abs/2308.12755). arXiv:2308.12755, 2023. Introduces Qualitative eXplainable Graphs for automated-driving scenes and motivates qualitative spatiotemporal reasoning as an interpretable scene representation.

### Qualitative Spatial And Temporal Reasoning

- A. G. Cohn and S. M. Hazarika. [Qualitative Spatial Representation and Reasoning: An Overview](https://doi.org/10.3233/FUN-2001-461-202). Fundamenta Informaticae, 46(1-2):1-29, 2001. Survey of qualitative spatial representation, including topology, distance, orientation, shape, and spatial change.
- Frank Dylla, Jae Hee Lee, Till Mossakowski, Thomas Schneider, Andre Van Delden, Jasper Van De Ven, and Diedrich Wolter. [A Survey of Qualitative Spatial and Temporal Calculi: Algebraic and Computational Properties](https://arxiv.org/abs/1606.00133). arXiv:1606.00133, 2016. Broad survey of qualitative calculi and their computational properties.
- James F. Allen. [Maintaining Knowledge about Temporal Intervals](https://dblp.org/rec/journals/cacm/Allen83). Communications of the ACM, 26(11):832-843, 1983. Defines Allen interval relations, a foundation for qualitative temporal reasoning.
- David A. Randell, Zhan Cui, and Anthony G. Cohn. [A Spatial Logic Based on Regions and Connection](https://philpapers.org/rec/RANASL). KR, pages 165-176, 1992. Introduces the Region Connection Calculus family used for qualitative topological relations.
- Nico Van de Weghe, Anthony G. Cohn, Guy De Tre, and Philippe De Maeyer. [A Qualitative Trajectory Calculus as a Basis for Representing Moving Objects in Geographical Information Systems](https://eudml.org/doc/209410). Control and Cybernetics, 35(1):97-119, 2006. Defines QTC-style qualitative relations for moving objects.
- Till Mossakowski and Reinhard Moratz. [Qualitative Reasoning about Relative Direction of Oriented Points](https://doi.org/10.1016/j.artint.2011.10.003). Artificial Intelligence, 180-181:34-45, 2012. Presents OPRA-style qualitative direction reasoning for oriented points.

## Validation

Run tests and linting before publishing changes:

```powershell
pytest
ruff check src tests
```

The core tests cover configuration loading, depth normalization, graph construction, serialization, tracker configuration, recording output, recording depth fallback, and QXG graph export.
