import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file

from asl_final_common import (
    HAND_CONNECTIONS,
    DEFAULT_LABELS,
    MODEL_PATH,
    extract_landmarks_from_hand,
    load_checkpoint,
    load_mediapipe_hands,
    rgb_frame_to_mp_image,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "train"


@dataclass
class SessionTracker:
    hands: object
    last_timestamp_ms: int = 0


def build_sample_index():
    sample_index = {}
    for label in DEFAULT_LABELS:
        class_dir = DATA_DIR / label
        if not class_dir.exists():
            continue
        candidates = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if candidates:
            sample_index[label] = candidates[0].resolve()
    return sample_index


def create_app(model_path, min_confidence):
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).with_name("web_templates")),
    )

    checkpoint, classifier = load_checkpoint(model_path)
    labels = checkpoint["labels"]
    mp, _ = load_mediapipe_hands(static_image_mode=True, max_num_hands=1)
    sample_index = build_sample_index()
    trackers = {}
    trackers_lock = Lock()

    def get_tracker(session_id):
        with trackers_lock:
            tracker = trackers.get(session_id)
            if tracker is None:
                _, hands = load_mediapipe_hands(static_image_mode=False, max_num_hands=1)
                tracker = SessionTracker(hands=hands)
                trackers[session_id] = tracker
            return tracker

    @app.get("/")
    def index():
        gallery = [
            {"label": label, "has_sample": label in sample_index}
            for label in DEFAULT_LABELS
        ]
        return render_template(
            "index.html",
            min_confidence=min_confidence,
            model_name=Path(model_path).name,
            gallery=gallery,
        )

    @app.get("/samples/<label>")
    def sample_image(label):
        label = label.upper()
        path = sample_index.get(label)
        if path is None:
            abort(404)
        return send_file(path)

    @app.post("/predict")
    def predict():
        if "frame" not in request.files:
            return jsonify({"error": "Missing frame upload."}), 400

        raw = request.files["frame"].read()
        image_np = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode image."}), 400
        session_id = request.form.get("session_id", "default")
        tracker = get_tracker(session_id)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = rgb_frame_to_mp_image(mp, rgb)

        detect_start = time.perf_counter()
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= tracker.last_timestamp_ms:
            timestamp_ms = tracker.last_timestamp_ms + 1
        tracker.last_timestamp_ms = timestamp_ms
        result = tracker.hands.detect_for_video(mp_image, timestamp_ms)
        detect_ms = (time.perf_counter() - detect_start) * 1000.0

        if not result.hand_landmarks:
            return jsonify(
                {
                    "hand_detected": False,
                    "prediction": None,
                    "confidence": 0.0,
                    "detect_ms": detect_ms,
                    "classify_ms": 0.0,
                    "accepted": False,
                    "landmarks": [],
                    "connections": HAND_CONNECTIONS,
                }
            )

        hand_landmarks = result.hand_landmarks[0]
        features = extract_landmarks_from_hand(result.hand_landmarks[0])
        if features is None:
            return jsonify(
                {
                    "hand_detected": False,
                    "prediction": None,
                    "confidence": 0.0,
                    "detect_ms": detect_ms,
                    "classify_ms": 0.0,
                    "accepted": False,
                    "landmarks": [],
                    "connections": HAND_CONNECTIONS,
                }
            )

        classify_start = time.perf_counter()
        probs = classifier.predict_proba(features.reshape(1, -1))[0]
        classify_ms = (time.perf_counter() - classify_start) * 1000.0
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        prediction = labels[pred_idx]

        return jsonify(
            {
                "hand_detected": True,
                "prediction": prediction,
                "confidence": confidence,
                "detect_ms": detect_ms,
                "classify_ms": classify_ms,
                "accepted": confidence >= min_confidence,
                "landmarks": [
                    {"x": float(point.x), "y": float(point.y), "z": float(point.z)}
                    for point in hand_landmarks
                ],
                "connections": HAND_CONNECTIONS,
            }
        )

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Run the ASL web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    app = create_app(args.model_path, args.min_confidence)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
