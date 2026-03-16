# Traffic Sign Recognition (CodeAlpha Task 04)

![Project Banner](assets/banner.svg)

> End-to-end traffic sign pipeline using YOLO-based detection and CNN-based classification, with cleaned project structure and compatibility entrypoints.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Core Features](#core-features)
3. [Current Repository Structure](#current-repository-structure)
4. [Implementation Summary](#implementation-summary)
5. [Environment & Dependencies](#environment--dependencies)
6. [How to Run](#how-to-run)
7. [Output & Metrics](#output--metrics)
8. [Validation Status](#validation-status)
9. [Author](#author)

---

## Project Overview

This project is designed to:

- Detect traffic signs from frames using YOLO inference.
- Classify detected sign regions into specific class labels.
- Produce annotated media outputs for visual inspection.

The codebase was refactored into a cleaner `src/` layout while preserving root-level command compatibility.

---

## Core Features

- **Clean architecture:** core model utilities separated from runnable scripts.
- **Backward compatibility:** root files delegate to the `src` package.
- **Safer execution path:** explicit checks for missing files and invalid frames.
- **Cross-version robustness:** improved OpenCV output-layer index handling.
- **Clear entrypoints:** separate scripts for video inference and annotation visualization.

---

## Current Repository Structure

```text
codealpha_ai_task04/
├── .gitignore
├── README.md
├── darknet.py                     # compatibility wrapper
├── recognition.py                 # compatibility wrapper (video)
├── recognition_images.py          # compatibility wrapper (image annotations)
├── recognition_videos.py          # compatibility wrapper (labeled image annotations)
└── src/
    ├── __init__.py
    ├── core/
    │   ├── __init__.py
    │   └── darknet.py             # Darknet/YOLO utility definitions
    └── scripts/
        ├── __init__.py
        ├── recognition.py         # main video pipeline
        ├── recognition_images.py  # draw annotation boxes by class id
        └── recognition_videos.py  # draw annotation boxes by class name
```

---

## Implementation Summary

- Refactored implementation into package-based layout under `src/core` and `src/scripts`.
- Added compatibility wrappers so previous run commands still work.
- Added/updated module headers with project metadata (Created: Aug 2024).
- Updated error handling to fail early with actionable messages when required assets are missing.
- Improved prediction path stability for empty crops and frame boundary conditions.

---

## Environment & Dependencies

### Python

- Python 3.9+

### Required packages

- `opencv-python`
- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow`
- `keras`
- `torch`

Install example:

```bash
pip install opencv-python numpy pandas matplotlib tensorflow keras torch
```

---

## How to Run

### 1) Video recognition

```bash
python recognition.py --video input_video/traffic-sign-to-test.mp4 --output result.mp4
```

Working demo (no external data/weights/model needed):

```bash
python recognition.py --demo --output result_demo.mp4
```

### 2) Visualize image annotations (class id)

```bash
python recognition_images.py --image-num 00001 --output example.png
```

Working demo (no dataset needed):

```bash
python recognition_images.py --demo --output example_demo.png
```

### 3) Visualize image annotations (class name)

```bash
python recognition_videos.py --image-num 00074 --output results3.png
```

Working demo (no dataset needed):

```bash
python recognition_videos.py --demo --output results3_demo.png
```

---

## Output & Metrics

### Output artifacts

- **Video pipeline:** annotated output video (default: `result.mp4`).
- **Image annotation script:** saved preview image (default: `example.png`).
- **Labeled annotation script:** saved preview image (default: `results3.png`).

### Runtime reporting (clean wording)

- **Frames processed:** total frames handled by the video pipeline.
- **Total processing time:** wall-clock inference duration.
- **Average throughput:** effective frames-per-second (FPS).

---

## Validation Status

- ✅ Python syntax validation passes (`python -m compileall .`).
- ✅ Entry-point wiring is consistent after refactor.
- ⚠️ Full runtime requires local assets and dependencies not committed to this repo:
  - `data/`
  - `weights/`
  - `model/`

Once those assets are present, the scripts run with the commands listed above.

---

## Author

**Uzair Ashfaq**  
Date: August 2024
Readme Updated: March 2026