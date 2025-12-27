import os 
import time
import pyaudio
import playsound
from gtts import gTTS
from model import maze_ai
import speech_recognition as sr


lang='en'


while True:
    def get_audio():
        print("function started")
        r=sr.Recognizer()
        with sr.Microphone() as source:
            audio=r.listen(source)
            said = ""
        said= r.recognize_google(audio)
        
        print(said)
        text = maze_ai(said)
        print(text)

    get_audio()