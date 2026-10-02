# Real-Time ASL Gesture Recognition: MediaPipe + LightGBM

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green?style=for-the-badge&logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-008080?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)

A high-speed, edge-optimized American Sign Language (ASL) recognition system. By decoupling spatial computer vision (via Google MediaPipe's lightweight 21-point 3D landmark extraction) from classification (via an optimized LightGBM gradient-boosted decision tree), this pipeline achieves **sub-10ms real-time inference latency** on standard consumer CPUs without requiring discrete GPUs.

---

## ⚡ Architecture Pipeline

```
  ┌─────────────┐       ┌────────────────────────┐       ┌──────────────────────┐       ┌────────────────┐
  │ Webcam Feed │ ───▶  │  MediaPipe HandLandmark │ ───▶  │ Normalized 63-dim    │ ───▶  │ LightGBM GBDT  │
  │ (30-60 FPS) │       │  (21 3D Spatial Nodes) │       │ Landmark Vector      │       │ Classifier     │
  └─────────────┘       └────────────────────────┘       └──────────────────────┘       └───────┬────────┘
                                                                                                │
                                                                                        Predicted Gesture
                                                                                        + Confidence Score
                                                                                                │
                                                                                                ▼
                                                                                        ┌────────────────┐
                                                                                        │  Flask Web UI  │
                                                                                        │  Real-Time HUD │
                                                                                        └────────────────┘
```

---

## 🚀 Key Advantages

- **Zero GPU Overhead**: Runs smoothly at 30+ FPS on ultrabooks and low-power hardware.
- **Microsecond Classification**: Gradient boosting on structured coordinates executes orders of magnitude faster than deep 2D/3D CNNs.
- **Rotation & Scale Invariance**: Features are normalized relative to wrist origin and palm span.

---

## 🛠️ Quickstart

### 1. Environment Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-final.txt
```

### 2. Run the Web Interface
```bash
python src/webapp.py
```
Open `http://localhost:5000` in your web browser.

---

## 📁 Repository Structure

```
├── data/                    # Landmark feature vectors
├── models/
│   ├── asl_landmark_lightgbm.pkl   # Serialized LightGBM classifier
│   └── hand_landmarker.task        # MediaPipe hand tracking model bundle
├── src/
│   ├── asl_final_common.py         # Normalization & preprocessing primitives
│   ├── webapp.py                   # Flask server with streaming video feed
│   └── web_templates/index.html    # Interactive HUD UI
├── requirements-final.txt
└── README.md
```

---

## 👤 Author
- **Ramani** ([@Ramani-21-05](https://github.com/Ramani-21-05))
- Final-Year Artificial Intelligence & Data Science Undergraduate
