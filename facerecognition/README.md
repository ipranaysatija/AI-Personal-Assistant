# Face Recognition

Local face auth for the assistant. Uses OpenCV's LBPH recognizer — no dlib, installs cleanly on Windows.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. **Enroll** a person (captures 30 webcam samples):
   ```bash
   python enroll.py Pranay
   ```
2. **Train** the model across everyone enrolled:
   ```bash
   python train.py
   ```
3. **Live demo**:
   ```bash
   python recognize.py
   ```
4. **One-shot auth** from code:
   ```python
   from facerecognition.auth import authenticate
   if authenticate(allowed=["Pranay"]):
       ...
   ```

## Files

- `enroll.py` — capture face samples into `faces/<name>/`
- `train.py` — build `model.yml` + `labels.json`
- `recognize.py` — live webcam overlay
- `auth.py` — `authenticate(allowed, timeout)` for gating the assistant

Lower `CONFIDENCE_THRESHOLD` in `auth.py` for stricter matches (LBPH: lower = better match).
