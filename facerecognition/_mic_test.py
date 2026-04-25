import sys
import speech_recognition as sr

device_index = int(sys.argv[1]) if len(sys.argv) > 1 else None

r = sr.Recognizer()
r.energy_threshold = 300
r.dynamic_energy_threshold = True

print(f"Using device_index={device_index} ({'default' if device_index is None else sr.Microphone.list_microphone_names()[device_index]})")

with sr.Microphone(device_index=device_index) as source:
    print("Calibrating for ambient noise (0.8s)...")
    r.adjust_for_ambient_noise(source, duration=0.8)
    print(f"Energy threshold now: {r.energy_threshold:.1f}")
    print("Speak loudly and clearly now (up to 6s)...")
    try:
        audio = r.listen(source, timeout=10, phrase_time_limit=6)
    except sr.WaitTimeoutError:
        print("No speech detected within 10s.")
        raise SystemExit

out = "E:/programing/Personal Ai Assistant/facerecognition/_captured.wav"
with open(out, "wb") as f:
    f.write(audio.get_wav_data())
print(f"Saved captured audio to {out}")

print("Sending to Google...")
try:
    text = r.recognize_google(audio)
    print(f"SUCCESS: {text}")
except sr.UnknownValueError:
    print("Google returned UnknownValueError.")
except sr.RequestError as e:
    print(f"Google API error: {e}")
