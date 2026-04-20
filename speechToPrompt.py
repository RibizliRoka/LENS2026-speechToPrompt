import os
import cv2 as cv
import numpy as np
import whisper
from google import genai
from PIL import Image
from google.genai import types
from dotenv import load_dotenv

import threading
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pynput import keyboard

startRecording_event = threading.Event()
stopRecording_event = threading.Event()
is_recording = False
audio_chunks = []

def on_press(key, injected):
    global is_recording
    try:
        #This is for the toggle on s, first if is to start recording and 
        #second if is end recording
        if key.char == 's' and not is_recording:
            is_recording = True
            startRecording_event.set()
        elif key.char == 's' and is_recording:
            is_recording = False
            stopRecording_event.set()

    except AttributeError:
        pass

def callback(indata, outdata, frames, time, status):
    global audio_chunks
    if status:
        print(status)
    audio_chunks.append(indata.copy())


def main():
    #record audio
    print("Press s to start, and then press s to stop recording")

    #Listener for Keyboard
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    #Start recording until s is clicked
    startRecording_event.wait()
    print("Starting recording.....")

    with sd.Stream(channels=2, callback=callback):
        #Stop recording when s is clicked again
        stopRecording_event.wait()

    full_audio = np.concatenate(audio_chunks)

    print("Stop recording.....")

    #Stop the listener
    listener.stop()

    # Save as WAV file
    fs = 44100  # Sample rate
    write("output_sd.wav", fs, full_audio)
    print("Audio saved to output_sd.wav")

    #whisper stuff
    model = whisper.load_model("tiny")
    result = model.transcribe("output_sd.wav")
    print(result["text"])

    os.environ["QT_QPA_PLATFORM"] = "xcb"
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("No camera")
        exit()

    while True:
        ret, frame = cam.read()
        if not ret:
            print("No frame")
            break
        
        cmd = input("q to quit, enter for picture: ").lower()
        if cmd != "q":
            cv.imwrite("output.png", frame)
            speechText = speechToText()
            speechText = "What is in the picture"
            ourPrompt = speechTextToAI(speechText)
            print(ourPrompt)
            # update txt file here
            print("got pic")
            os.remove("output.png")
        else:
            break
    cam.release()

def speechTextToAI(speechText):
    load_dotenv()
    client = genai.Client(api_key=os.getenv("apiKey"))
    img = Image.open("/home/laura-szabo/Code/School/Lens/speechToPrompt/output.png")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
        system_instruction="An image will be passed in along with a prompt asking the robot to pick a certain object or set of " \
        "objects up, however the robot's language model only understands very basic instructions like \'pick up the red block\'. What you need to" \
        "do is take the following prompt, and simplify it down for the robot using the picture as reference. Only return the modified prompt and nothing else."),
        contents=[img, speechText],
    )
    return response.text    


def speechToText(): #
    pass

if __name__ == "__main__":
    main()