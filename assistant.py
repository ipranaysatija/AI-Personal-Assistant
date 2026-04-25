import speech_recognition as sr

from model import maze_ai
from facerecognition.auth import recognize_once

ALLOWED_USERS = ["Pranay"]


def get_audio(recognizer: sr.Recognizer) -> str:
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.6)
        audio = recognizer.listen(source, timeout=8, phrase_time_limit=8)
    return recognizer.recognize_google(audio)


def main() -> None:
    print("Verifying face...")
    name, conf = recognize_once(timeout=15)
    if name is None or name not in ALLOWED_USERS:
        print("Face not recognized. Access denied.")
        return
    print(f"Welcome, {name} (confidence={conf:.1f}).")

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    while True:
        print("Listening...")
        try:
            said = get_audio(recognizer)
        except sr.WaitTimeoutError:
            print("No speech detected. Try again.")
            continue
        except sr.UnknownValueError:
            print("Didn't catch that.")
            continue
        except sr.RequestError as e:
            print(f"Speech API error: {e}")
            return
        print(f"You: {said}")
        print(f"AI: {maze_ai(said)}")


if __name__ == "__main__":
    main()
