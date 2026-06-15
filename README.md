# 🚗 Vehicle Detection & Tracking (YOLOv8 + DeepSORT)

A computer vision pipeline that detects vehicles in video footage and tracks each one across frames with a persistent ID — built with **YOLOv8** for detection and **DeepSORT** for multi-object tracking.

---

## Overview

This project trains a custom YOLOv8s object detector on a 5-class vehicle dataset, then feeds its detections into a DeepSORT tracker to produce smooth, ID-consistent bounding boxes over an entire video — the kind of pipeline used in traffic monitoring, smart-city analytics, and autonomous driving perception stacks.

**Pipeline:**

```
Video frames → YOLOv8 detector → filter target classes → DeepSORT tracker → annotated output video
```

---

## ✨ Features

- Custom-trained **YOLOv8s** detector for 5 vehicle classes
- Real-time-capable **DeepSORT** tracking (`deep_sort_realtime`) with persistent track IDs
- Per-track color coding for easy visual differentiation
- Smart on-frame label placement (auto-flips label box near frame edges)
- End-to-end `ObjectTracker` class — plug in any YOLO model + class list
- Full video-in → video-out processing with progress logging

---

## 📦 Dataset

5-class vehicle detection dataset (Ambulance, Bus, Car, Motorcycle, Truck):

- **Dataset + sample output:** [Google Drive folder](https://drive.google.com/drive/folders/1bk7FUKeH1y4cJkduXq2O63nM77y22d1Z?usp=sharing)
- **Validation set:** 250 images / 454 annotated instances

---

## 🏋️ Model Training

| Setting | Value |
|---|---|
| Base model | `yolov8s.pt` |
| Image size | 640 |
| Epochs | 55 |
| Batch size | 16 |
| Hardware | Tesla T4 (Colab) |
| Params / GFLOPs | 11.1M / 28.4 |

```bash
yolo task=detect mode=train model=yolov8s.pt \
  data=dataset.yaml \
  imgsz=640 epochs=55 batch=16 \
  project=Training_RES name=VehiclesDetection
```

### Validation Results

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| **All** | 250 | 454 | 0.695 | 0.545 | 0.613 | 0.450 |
| Ambulance | 50 | 64 | 0.822 | 0.750 | 0.813 | 0.693 |
| Bus | 30 | 46 | 0.724 | 0.696 | 0.706 | 0.571 |
| Car | 90 | 238 | 0.712 | 0.445 | 0.511 | 0.341 |
| Motorcycle | 42 | 46 | 0.649 | 0.565 | 0.593 | 0.341 |
| Truck | 38 | 60 | 0.570 | 0.267 | 0.440 | 0.304 |

**Inference speed:** ~10.6ms/frame (T4 GPU) — preprocess 2.4ms, inference 9.6ms, postprocess 1.3ms.

> 📝 *Car and Truck recall are the weakest spots — likely dataset imbalance/occlusion driven, and a good target for future augmentation or more training data.*

---

## 🎯 Object Tracking (DeepSORT)

The `ObjectTracker` class wraps a trained YOLO model with a `DeepSort` tracker:

- Filters raw detections down to a configurable list of target classes (`Ambulance`, `Bus`, `Car`, `Motorcycle`, `Truck`)
- Converts YOLO `xyxy` boxes to the `[x, y, w, h]` format DeepSORT expects
- Assigns each confirmed track a random, persistent color for its lifetime
- Draws a class name + track ID label that flips to stay inside the frame
- Processes a full video file frame-by-frame and writes the annotated result with OpenCV's `VideoWriter`

```python
loaded_yolo = YOLO("path/to/best.pt")
target_classes = ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"]

object_tracker = ObjectTracker(loaded_yolo, target_classes, max_age=5, gpu=True)
object_tracker.process_video(video_path, output_dir)
```

**Test run:** processed a 2,362-frame traffic video end-to-end, exporting `object_track_video.mp4` with live progress logging every 400 frames.

---

## 🛠️ Tech Stack

- **Detection:** YOLOv8 (Ultralytics)
- **Tracking:** DeepSORT (`deep_sort_realtime`)
- **Core libs:** PyTorch, OpenCV
- **Environment:** Google Colab (T4 GPU)

---

## 🚀 Usage

```bash
# 1. Install dependencies
pip install ultralytics deep_sort_realtime

# 2. Train (or use the provided best.pt weights)
yolo task=detect mode=train model=yolov8s.pt data=dataset.yaml imgsz=640 epochs=55 batch=16

# 3. Run detection + tracking on a video
python -c "
from ultralytics import YOLO
model = YOLO('best.pt')
tracker = ObjectTracker(model, ['Ambulance','Bus','Car','Motorcycle','Truck'], max_age=5, gpu=True)
tracker.process_video('input.mp4', 'output/')
"
```

---


## 📄 License

This project is for educational and portfolio purposes.
DATA & OUPTUT: https://drive.google.com/drive/folders/1F3Dby3p-5vEuQBX3evv-T87jXXHY8RnV?usp=drive_link
