import os
import cv2 as cv
import numpy as np
import whisper
from google import genai
from PIL import Image
from google.genai import types
from dotenv import load_dotenv


import logging
import threading
import time
import queue
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pynput import keyboard


#wake word stuff
import openwakeword
from openwakeword.model import Model


startRecording_event = threading.Event()
stopRecording_event = threading.Event()
is_recording = False
WWaudio_chunks = queue.Queue()
RECaudio_chunks = []


def listenForWakeWord():
   global is_recording
   model = Model(wakeword_model_paths=["/home/laura-szabo/Code/School/Lens/speechToPrompt/Hey_Grip_ee.onnx"])
   while True:
       frame = WWaudio_chunks.get()
       if frame is None:
           break


       prediction = model.predict(frame.flatten())
       score = list(prediction.values())[0]
      
       #prediction = model.predict(frame.flatten())
       #score = prediction.get("alexa_v0.1", 0)
       print(score)


       if score > 0.5 and not is_recording:
           is_recording = True
           startRecording_event.set()
           print("alexa start") 
           model.reset()
           while not WWaudio_chunks.empty():
               try:
                   WWaudio_chunks.get_nowait()
               except queue.Empty:
                   break
           time.sleep(3)
       elif score > 0.5 and is_recording:
           is_recording = False
           stopRecording_event.set()
           print("alexa stop")
           model.reset()
           break


def callback(indata, outdata, frames, time, status):
   global WWaudio_chunks, RECaudio_chunks
   WWaudio_chunks.put(indata.copy())
   if is_recording:
       RECaudio_chunks.append(indata.copy())


def speechToPromptMain():
   global RECaudio_chunks
   with sd.Stream(samplerate=16000,channels=1,dtype='int16',blocksize=1280,callback=callback):
       startRecording_event.wait()
       startRecording_event.clear()
       RECaudio_chunks.clear()
       print("recording")
       stopRecording_event.wait()
       stopRecording_event.clear()
   full_audio = np.concatenate(RECaudio_chunks)


   # Save as WAV file
   fs = 16000
   write("output_sd.wav", fs, full_audio)
   print("Audio saved to output_sd.wav")


   #whisper stuff
   model = whisper.load_model("tiny")
   result = model.transcribe("output_sd.wav")
   speechText = result["text"]
   print(speechText)


   os.environ["QT_QPA_PLATFORM"] = "xcb"
   os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
   cam = cv.VideoCapture(0)
   if not cam.isOpened():
       print("No camera")
       exit()


   ret, frame = cam.read()
   if not ret:
       print("No frame")
   else:
       cv.imwrite("output.png", frame)
       print("got pic")
       ourPrompt = speechTextToAI(speechText)
       print(ourPrompt)
       os.remove("output.png")
       os.remove("output_sd.wav")
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
       "do is take the following prompt, and simplify it down for the robot using the picture as reference. Only return the modified prompt and nothing else. Also ignore any hey grippy that you hear."),
       contents=[img, speechText],
   )
   return response.text   


if __name__ == "__main__":
   threading.Thread(target=listenForWakeWord, daemon=True).start()
   speechToPromptMain()
