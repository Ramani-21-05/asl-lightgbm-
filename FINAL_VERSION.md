# ASL LightGBM Web App

This project is trimmed down to a single runtime pipeline:

- Hand detection and tracking: `MediaPipe Hands`
- Classifier: `LightGBM` on normalized 21-point hand landmarks
- App entrypoint: `src/webapp.py`

## Supported Python

Use Python 3.12 for this app. MediaPipe support is typically available for CPython 3.9-3.12.

## Install

```bash
python3.12 -m venv .venv-final
source .venv-final/bin/activate
python -m pip install -r requirements-final.txt
```

## Run

```bash
python src/webapp.py
```

Then open `http://127.0.0.1:5000`.

## Required model files

The web app expects these files:

- `models/asl_landmark_lightgbm.pkl`
- `models/hand_landmarker.task`
