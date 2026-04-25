import os
import json
import cv2
import numpy as np

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "faces")
MODEL_PATH = os.path.join(BASE, "model.yml")
LABELS_PATH = os.path.join(BASE, "labels.json")


def train() -> None:
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"No enrollments found at {DATA_DIR}. Run enroll.py first.")

    images, labels, label_map = [], [], {}
    for idx, name in enumerate(sorted(os.listdir(DATA_DIR))):
        person_dir = os.path.join(DATA_DIR, name)
        if not os.path.isdir(person_dir):
            continue
        label_map[idx] = name
        for fname in os.listdir(person_dir):
            img = cv2.imread(os.path.join(person_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            images.append(img)
            labels.append(idx)

    if not images:
        raise SystemExit("No training images found.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    recognizer.write(MODEL_PATH)

    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f)

    print(f"Trained on {len(images)} images across {len(label_map)} people.")
    print(f"Model: {MODEL_PATH}")


if __name__ == "__main__":
    train()
