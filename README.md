# Accident Detection with YOLOv11

Real-time traffic accident detection system trained with YOLOv11 and deployed on edge hardware (Radxa CM5 / NVIDIA Jetson). Detects accidents in dashcam footage with a temporal confidence buffer and cooldown mechanism to reduce false positives.

---

## Demo

### Desktop inference on video (`live_detect_pth_save.py`)

[![Desktop inference demo](https://img.youtube.com/vi/gaxsCEG30dc/maxresdefault.jpg)](https://youtu.be/gaxsCEG30dc)

> Detection running on a recorded dashcam video using the `.pt` model on a Windows PC.

---

### Edge deployment on Radxa CM5 - live camera + accident video on screen

These clips were recorded directly from the Radxa CM5 deployment. A Radxa Camera 4K (IMX415, CSI) captured a screen playing accident footage while the RKNN-converted model ran on the NPU in real time.

<table>
  <tr>
    <td align="center">
      <video src="upload_video/event.mp4" controls width="360"></video><br/>
      <sub>Event 1</sub>
    </td>
    <td align="center">
      <video src="upload_video/event1.mp4" controls width="360"></video><br/>
      <sub>Event 2</sub>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <video src="upload_video/event2.mp4" controls width="360"></video><br/>
      <sub>Event 3</sub>
    </td>
  </tr>
</table>

---

## Overview

This project implements an end-to-end accident detection pipeline:

1. **Train** a YOLOv11s model with a two-stage frozen/unfrozen strategy at 1024×1024 resolution.
2. **Convert** the trained `.pt` model to ONNX then RKNN format for NPU inference.
3. **Run inference** on a PC (`.pt` model) or on edge hardware.
4. **Deploy** on Radxa CM5 using an RKNN-converted model that runs on the built-in NPU, with event recording, LED indicators, and an HTTP live stream.

---

## Repository Structure

```
├── train_yolo11.py                      # Two-stage YOLOv11 training script
├── live_detect_pth.py                   # Desktop inference on video files (.pt model)
├── live_detect_pth_save.py              # Desktop inference with output video saving
├── live_detect_pth_camera_jetson.py     # Jetson inference with USB camera (.pt model)
├── convert_rknn/                        # Model conversion pipeline
│   ├── convert_pth_to_onnx.py           #   Export .pt → .onnx
│   ├── convert_onnx_to_rknn.py          #   Convert .onnx → .rknn
│   ├── convert_onnx_to_rknn_optim.py    #   Optimized RKNN conversion
│   └── create_rknn_dataset.py           #   Build quantization dataset
├── live_detect_rknn_final/              # Full edge deployment package (Radxa CM5)
│   ├── live_detect_rknn_final.py        #   Main entry point
│   ├── detector.py                      #   RKNN inference + temporal logic
│   ├── camera.py                        #   Camera reader thread
│   ├── recorder.py                      #   Pre/post-buffer event recorder
│   ├── web_server.py                    #   MJPEG HTTP streaming server
│   ├── led.py                           #   GPIO LED indicator
│   ├── sdcard.py                        #   SD card path resolution
│   ├── writer.py                        #   Async image writer
│   └── config.py                        #   All runtime configuration
└── upload_video/                        # Sample output — Radxa CM5 edge deployment
    ├── event.mp4
    ├── event1.mp4
    └── event2.mp4
```

---

## Training (`train_yolo11.py`)

Training uses a **two-stage strategy** to prevent early overfitting:

| Stage | Epochs | Backbone | LR | Image size |
|-------|--------|----------|----|------------|
| 1 — Frozen warm-up | 30 | Frozen (first 8 layers) | 5e-4 | 1024 |
| 2 — Full fine-tuning | 80 | All layers unfrozen | 1e-4 | 1024 |

**Run training:**
```bash
python train_yolo11.py
```

Expects `dataset/dataset.yaml`. Outputs checkpoints to `runs/train/`.

---

## Model Conversion (`convert_rknn/`)

To deploy on Radxa CM5, the trained `.pt` model must be converted to RKNN format:

```bash
# Step 1 — Export to ONNX
python convert_rknn/convert_pth_to_onnx.py

# Step 2 — Convert ONNX to RKNN
python convert_rknn/convert_onnx_to_rknn.py
```

Use `create_rknn_dataset.py` to build a calibration dataset for INT8 quantization.

---

## Desktop Inference

### On a video file — with saving (`live_detect_pth_save.py`)

```bash
python live_detect_pth_save.py
```

Runs detection and saves the annotated output as a video file.

### On a video file — display only (`live_detect_pth.py`)

```bash
python live_detect_pth.py
```

Edit the `model_path` and `video_path` variables inside the script.

**Detection logic:**
- Runs YOLO on every 3rd frame (configurable `SKIP_FRAMES`)
- Draws bounding boxes for all predictions; red if ≥ confidence threshold, gray otherwise
- Accident is declared if:
  - Any box confidence ≥ `high_confidence` (0.7) — immediate trigger, or
  - Majority vote over last 3 frames, or
  - Within cooldown window after a previous trigger

### On a live USB camera (NVIDIA Jetson)

```bash
python live_detect_pth_camera_jetson.py
```

Opens `/dev/video0` via V4L2 (MJPEG) at 1280×720@30 fps. Falls back to GStreamer if needed.

---

## Edge Deployment — Radxa CM5 (`live_detect_rknn_final/`)

The full deployment package for Radxa CM5 uses a model converted to RKNN format (`.rknn`) that runs on the built-in NPU.

### Features

| Feature | Details |
|---|---|
| Model | YOLOv11s converted to `.rknn` (NPU inference) |
| Camera | Radxa Camera 4K (IMX415) via GStreamer / USB camera via V4L2 |
| Event recording | Saves annotated clips with pre/post buffer to SD card |
| LED indicator | GPIO LED — startup sequence + blink on accident |
| HTTP live stream | MJPEG stream served on port 8000 |
| Configurable threshold | Persisted in `/etc/ondai/threshold.txt` |
| Auto-restart | Designed to run as a `systemd` service |

### Run

```bash
cd live_detect_rknn_final
python live_detect_rknn_final.py
```

### Configuration (`config.py`)

Key variables you may want to adjust:

```python
MODEL_PATH        = 'yolov11s_epoch26.rknn'   # path to your RKNN model
CONF_THRESHOLD    = 0.5                        # accident confidence threshold
INPUT_SIZE        = 640                        # model input resolution
SKIP_FRAMES       = 2                          # infer every N+1 frames
PREBUFFER_SEC     = 1.0                        # seconds to save before event
POSTBUFFER_SEC    = 1.0                        # seconds to save after event
ENABLE_STREAM     = 1                          # HTTP MJPEG stream (0 to disable)
STREAM_PORT       = 8000
```

Environment variables override config values at runtime:
```bash
CONF_THRESHOLD=0.6 ENABLE_STREAM=0 python live_detect_rknn_final.py
```

---

## Requirements

### Training / Desktop
```
ultralytics
torch
opencv-python
numpy
```

### Edge (Radxa CM5)
```
rknn-toolkit-lite2   # for RKNN inference
opencv-python
numpy
```

---

## How It Works

```
Camera / Video
      │
      ▼
 Frame Reader  ──────────────────────────────►  Frame Queue
                                                      │
                                                      ▼
                                              Inference (every Nth frame)
                                                      │
                                              ┌───────┴────────┐
                                              │ Temporal Logic  │
                                              │  • High-conf   │
                                              │  • Majority    │
                                              │  • Cooldown    │
                                              └───────┬────────┘
                                                      │
                              ┌───────────────────────┼──────────────────────┐
                              ▼                       ▼                      ▼
                      Bounding box HUD       Event Recorder           LED indicator
                      (on-screen display)    (pre/post buffer         (GPIO blink)
                                             → SD card clips)
                                                      │
                                               HTTP MJPEG stream
                                               (web browser view)
```
