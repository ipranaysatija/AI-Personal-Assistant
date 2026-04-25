import os
import json
import time
import cv2

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "model.yml")
LABELS_PATH = os.path.join(BASE, "labels.json")
CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

CONFIDENCE_THRESHOLD = 70.0
TIMEOUT_SECONDS = 10


def _load():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        raise RuntimeError("Model not trained. Run enroll.py then train.py first.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH) as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    return recognizer, labels


def recognize_once(timeout: float = TIMEOUT_SECONDS):
    """Return (name, confidence) if a known face is seen within timeout, else (None, None)."""
    recognizer, labels = _load()
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Could not open webcam")

    deadline = time.time() + timeout
    result = (None, None)
    try:
        while time.time() < deadline:
            ok, frame = cam.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = CASCADE.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
            for (x, y, w, h) in faces:
                face = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                label_id, conf = recognizer.predict(face)
                if conf < CONFIDENCE_THRESHOLD:
                    result = (labels.get(label_id), conf)
                    return result
    finally:
        cam.release()
    return result


def authenticate(allowed: list[str] | None = None, timeout: float = TIMEOUT_SECONDS) -> bool:
    """True if a recognized face (optionally restricted to `allowed`) is seen in time."""
    name, _ = recognize_once(timeout=timeout)
    if name is None:
        return False
    if allowed is None:
        return True
    return name in allowed


if __name__ == "__main__":
    name, conf = recognize_once()
    if name:
        print(f"Recognized: {name} (confidence={conf:.1f})")
    else:
        print("No known face detected.")
