import json
import pickle
from collections import Counter, deque
from pathlib import Path

import numpy as np


DEFAULT_LABELS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
MODEL_PATH = Path("models/asl_landmark_lightgbm.pkl")
LANDMARK_DIM = 21 * 3
HAND_LANDMARKER_TASK_PATH = Path("models/hand_landmarker.task")
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class PredictionSmoother:
    def __init__(self, maxlen=8):
        self.labels = deque(maxlen=maxlen)
        self.confidences = deque(maxlen=maxlen)

    def push(self, label, confidence):
        self.labels.append(label)
        self.confidences.append(float(confidence))

    def top_label(self):
        if not self.labels:
            return "-"
        return Counter(self.labels).most_common(1)[0][0]

    def mean_confidence(self):
        if not self.confidences:
            return 0.0
        return float(sum(self.confidences) / len(self.confidences))


def ensure_model_dir(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def normalize_landmarks(landmarks):
    points = np.asarray(landmarks, dtype=np.float32).reshape(21, 3)
    wrist = points[0].copy()
    centered = points - wrist
    scale = np.linalg.norm(centered, axis=1).max()
    if scale < 1e-6:
        return None
    normalized = centered / scale
    return normalized.reshape(-1).astype(np.float32)


def extract_landmarks_from_hand(hand_landmarks):
    raw = []
    for landmark in hand_landmarks:
        raw.extend([landmark.x, landmark.y, landmark.z])
    return normalize_landmarks(raw)


def rgb_frame_to_mp_image(mp, rgb_frame):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


def load_mediapipe_hands(static_image_mode, max_num_hands=1):
    try:
        import mediapipe as mp
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MediaPipe is required for the landmark pipeline. "
            "Use a Python 3.9-3.12 environment and install mediapipe there."
        ) from exc

    model_path = HAND_LANDMARKER_TASK_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Hand landmarker model not found at {model_path}. "
            "Download the official MediaPipe hand landmarker task file first."
        )

    running_mode = mp.tasks.vision.RunningMode.IMAGE if static_image_mode else mp.tasks.vision.RunningMode.VIDEO
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
        num_hands=max_num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hands = mp.tasks.vision.HandLandmarker.create_from_options(options)
    return mp, hands


def save_checkpoint(path, classifier, labels, train_stats):
    path = ensure_model_dir(path)
    payload = {
        "classifier": classifier,
        "labels": list(labels),
        "input_dim": LANDMARK_DIM,
        "train_stats": dict(train_stats),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_checkpoint(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {path}. Place the LightGBM checkpoint there before starting src/webapp.py."
        )
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    return checkpoint, checkpoint["classifier"]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
