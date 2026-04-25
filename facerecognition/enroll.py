import os
import sys
import cv2

DATA_DIR = os.path.join(os.path.dirname(__file__), "faces")
CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
SAMPLES_PER_PERSON = 30


def enroll(name: str) -> None:
    person_dir = os.path.join(DATA_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Could not open webcam")

    print(f"Capturing {SAMPLES_PER_PERSON} samples for '{name}'. Look at the camera.")
    count = 0
    while count < SAMPLES_PER_PERSON:
        ok, frame = cam.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

        for (x, y, w, h) in faces[:1]:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))
            cv2.imwrite(os.path.join(person_dir, f"{count:03d}.png"), face)
            count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{count}/{SAMPLES_PER_PERSON}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

        cv2.imshow("Enroll - press q to abort", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} samples to {person_dir}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else input("Name: ").strip()
    if not name:
        raise SystemExit("Name required")
    enroll(name)
