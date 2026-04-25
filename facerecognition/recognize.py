import os
import json
import cv2

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "model.yml")
LABELS_PATH = os.path.join(BASE, "labels.json")
CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

CONFIDENCE_THRESHOLD = 70.0


def live() -> None:
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        raise SystemExit("Model not trained. Run enroll.py then train.py first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH) as f:
        labels = {int(k): v for k, v in json.load(f).items()}

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise SystemExit("Could not open webcam")

    while True:
        ok, frame = cam.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            label_id, conf = recognizer.predict(face)
            if conf < CONFIDENCE_THRESHOLD:
                text = f"{labels.get(label_id, '?')} ({conf:.0f})"
                color = (0, 255, 0)
            else:
                text = f"unknown ({conf:.0f})"
                color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Recognition - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live()
